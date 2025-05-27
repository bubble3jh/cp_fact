import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
import argparse
import os
# run command : python generate_qwen_jsonl.py --fact_type nonfactual
def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
    device_map="auto",
    torch_dtype=torch.float16,  # 확실하게 fp16으로
    trust_remote_code=True
    )
    model.eval()

    input_path = os.path.join(args.base_path, "prompts", f"fever_{args.fact_type}_final.jsonl")
    if args.n_lines != -1:
        output_path = os.path.join(args.base_path, "outputs", f"{args.fact_type}_{args.model_name.split('/')[-1].lower().replace('-', '_')}_nlines{args.n_lines}.jsonl")
    else:
        output_path = os.path.join(args.base_path, "outputs", f"{args.fact_type}_{args.model_name.split('/')[-1].lower().replace('-', '_')}.jsonl")

    with open(input_path, "r") as infile:
        # 10 줄에서 cut
        prompt_data = [json.loads(line.strip()) for line in infile]
        if args.n_lines != -1:
            prompt_data = prompt_data[:args.n_lines]

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="/data3/bubble3jh/FactualityPrompt/")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--fact_type", type=str, default="factual", choices=["factual", "nonfactual"])
    parser.add_argument("--n_lines", type=int, default=10)
    args = parser.parse_args()
    main(args)