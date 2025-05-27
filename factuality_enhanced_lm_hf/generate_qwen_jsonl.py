import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
import argparse
import os

# run command: python generate_qwen_jsonl.py --fact_type nonfactual --n_lines 2000 --offset 4000

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.float16,  # 확실하게 fp16으로
        trust_remote_code=True
    )
    model.eval()

    # ----- 입력 파일 경로 -----
    input_path = os.path.join(args.base_path, "prompts", f"fever_{args.fact_type}_final.jsonl")

    # ----- 프롬프트 불러오기 & offset 슬라이싱 -----
    with open(input_path, "r") as infile:
        all_prompts = [json.loads(line.strip()) for line in infile]
        if args.n_lines != -1:
            prompt_data = all_prompts[args.offset : args.offset + args.n_lines]
        else:
            prompt_data = all_prompts[args.offset :]

    # ----- 출력 파일명 생성 -----
    fact_type = args.fact_type
    model_id = args.model_name.split('/')[-1].lower().replace('-', '_')
    offset_str = f"offset{args.offset:04d}"
    nlines_str = f"n{args.n_lines}" if args.n_lines != -1 else "nall"

    output_dir = os.path.join(args.base_path, "outputs", "noncp", fact_type, model_id)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{offset_str}_{nlines_str}.jsonl")

    # ----- 생성 루프 -----
    with open(output_path, "w") as outfile, torch.inference_mode():
        for ex in tqdm(prompt_data, desc=f"Generating {args.model_name} for {args.fact_type}"):
            prompt = ex["prompt"]
            input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **input_ids,
                max_new_tokens=60,
                temperature=1.0,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id
            )

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            continuation = generated[len(prompt):].strip()

            json.dump({"prompt": prompt, "text": continuation}, outfile)
            outfile.write("\n")

    print(f"[DONE] Output saved to → {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="/data3/bubble3jh/FactualityPrompt/")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--fact_type", type=str, default="factual", choices=["factual", "nonfactual"])
    parser.add_argument("--n_lines", type=int, default=10, help="생성할 라인 수")
    parser.add_argument("--offset", type=int, default=0, help="시작 인덱스 (슬라이스 오프셋)")
    args = parser.parse_args()
    main(args)
