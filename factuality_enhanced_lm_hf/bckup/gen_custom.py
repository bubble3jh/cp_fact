import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

input_path = "/data3/bubble3jh/FactualityPrompt/prompts/fever_factual_final.jsonl"
output_path = "/data3/bubble3jh/FactualityPrompt/outputs/factual_qwen3_0.6b.jsonl"

with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
    for line in tqdm(infile):
        prompt_obj = json.loads(line)
        prompt = prompt_obj['prompt']

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, top_p=0.9, temperature=1.0)

        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        continuation = gen_text[len(prompt):]  # 잘라내기

        json.dump({
            "prompt": prompt,
            "text": continuation.strip()
        }, outfile)
        outfile.write("\n")
