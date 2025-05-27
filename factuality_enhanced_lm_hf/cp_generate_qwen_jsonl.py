#!/usr/bin/env python
"""
Conformal-Prediction constrained generation for FactualityPrompt.
 1) τ (cumulative-prob) calibration on Wikitext-2
 2) CP-constrained token-by-token generation
 3) save JSONL -> evaluate_v3_final.py 호환
"""

import json, os, argparse, random, pathlib, pickle, sys
import torch, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ------------------ USER-CONFIG ------------------
RNG_SEED = 42
random.seed(RNG_SEED); torch.manual_seed(RNG_SEED)
# -------------------------------------------------

def calibrate_tau(args, model, tokenizer, device):
    """누적확률 τ 계산 → pickled 파일 캐시"""
    TAU_CACHE = os.path.join(
        "./caches",
        f"{args.model_name.split('/')[-1].lower().replace('-', '_')}_alpha{args.alpha}_wiki2_n_calib{args.n_calib}.pkl"
    )
    if os.path.exists(TAU_CACHE):
        with open(TAU_CACHE, "rb") as f:
            tau = pickle.load(f)
        print(f"[τ] loaded from cache: {tau:.4f}")
        return tau

    print(f"[τ] calibrating on Wikitext-2 ({args.n_calib} sents, α={args.alpha}) …")
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")[:args.n_calib]
    masses = []

    model.eval()
    with torch.no_grad():
        for line in tqdm(wiki["text"]):
            if not line.strip():
                continue
            ids = tokenizer(line, return_tensors="pt").to(device)
            ctx, labels = ids["input_ids"][:, :-1], ids["input_ids"][:, 1:]

            logits = model(ctx).logits[0]  # (seq_len, vocab)
            for j, y in enumerate(labels[0]):
                logit_vec = logits[j]
                prob = torch.softmax(logit_vec, dim=-1)
                sorted_p, sorted_idx = torch.sort(prob, descending=True)
                cum = torch.cumsum(sorted_p, 0)
                idx = (sorted_idx == y).nonzero(as_tuple=True)[0].item()
                masses.append(cum[idx].item())

    tau = float(np.quantile(masses, 1 - args.alpha))
    with open(TAU_CACHE, "wb") as f:
        pickle.dump(tau, f)
    print(f"[τ] calibrated: {tau:.4f}  (saved → {TAU_CACHE})")
    return tau

def cp_constrained_generate(args, prompt, model, tokenizer, tau, device):
    """가장 단순한 CP-제약: 누적≤τ 집합에서 greedy 선택"""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        for _ in range(args.max_new_tok):
            logits = model(ids).logits[:, -1, :]
            probs  = torch.softmax(logits, -1).squeeze()
            sorted_p, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_p, 0)
            allowed = sorted_idx[cum <= tau]

            if allowed.numel() == 0:
                break  # τ가 너무 작아서 집합이 비어 있음

            next_id = allowed[0].unsqueeze(0)  # greedy
            ids = torch.cat([ids, next_id.unsqueeze(0)], dim=-1)

            if next_id.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(ids[0], skip_special_tokens=True)

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    device = model.device

    # ----- τ 준비 -----
    tau = calibrate_tau(args, model, tokenizer, device)

    # ----- 프롬프트 읽기 -----
    input_path = os.path.join(args.base_path, "prompts", f"fever_{args.fact_type}_final.jsonl")
    with open(input_path) as f:
        all_lines = [json.loads(l) for l in f]
        if args.n_lines != -1:
            prompts = all_lines[args.offset : args.offset + args.n_lines]
        else:
            prompts = all_lines[args.offset :]

    # ----- 출력 경로 생성 -----
    fact_type = args.fact_type
    model_id = args.model_name.split('/')[-1].lower().replace('-', '_')
    offset_str = f"offset{args.offset:04d}"
    nlines_str = f"n{args.n_lines}" if args.n_lines != -1 else "nall"

    # 경로: outputs/cp/factual/qwen3_0.6b/offset0000_n2000.jsonl
    output_dir = os.path.join(args.base_path, "outputs", "cp", fact_type, model_id)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{offset_str}_{nlines_str}.jsonl")

    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # ----- 생성 -----
    model.eval()
    with open(out_path, "w") as fout, torch.no_grad():
        for item in tqdm(prompts, desc=f"CP-Generating {args.model_name}"):
            full_text = cp_constrained_generate(args, item["prompt"], model, tokenizer, tau, device)
            continuation = full_text[len(item["prompt"]):].strip()
            json.dump({"id": item["id"], "prompt": item["prompt"], "text": continuation}, fout, ensure_ascii=False)
            fout.write("\n")

    print(f"[DONE] saved → {out_path}")
    print(">> 이제 평가 예시:")
    print(f"PYTHONPATH=. python src/evaluate_v3_final.py "
          f"--prompt_type {args.fact_type} "
          f"--gen_path {out_path}")

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--base_path", type=str, default="/data3/bubble3jh/FactualityPrompt")
    pa.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    pa.add_argument("--fact_type", type=str, choices=["factual", "nonfactual"], default="factual")
    pa.add_argument("--n_lines", type=int, default=10)
    pa.add_argument("--offset", type=int, default=0, help="데이터 시작 위치 오프셋")
    pa.add_argument("--n_calib", type=int, default=500)
    pa.add_argument("--alpha", type=float, default=0.05)
    pa.add_argument("--max_new_tok", type=int, default=60)
    args = pa.parse_args()
    main(args)
