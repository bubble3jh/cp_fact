# 🧠 Factuality Evaluation Pipeline (Qwen-based)

This repository combines two components:

1. `factuality_enhanced_lm_hf/` – For **text generation** using Qwen models (with/without CP)
2. `FactualityPrompt/` – For **factuality evaluation** of generated outputs using GPT-based metrics

---

## 📦 Directory Structure

```
.
├── factuality_enhanced_lm_hf/
│   ├── generate_qwen_jsonl.py
│   ├── cp_generate_qwen_jsonl.py
│   └── qwen_gen.sh
├── FactualityPrompt/
│   └── src/
│       └── evaluate_v3_final.py
```

---

## 🚀 Step 1: Generation

To generate text using Qwen models across different settings (model size × factuality type × CP/non-CP), run:

```bash
cd factuality_enhanced_lm_hf

bash qwen_gen.sh
```

This script launches parallel jobs on GPUs 0–7 using `CUDA_VISIBLE_DEVICES` and saves results in:

```
../FactualityPrompt/outputs/{cp,noncp}/{factual,nonfactual}/{model_name}/offsetXXXX_n2000.jsonl
```

Example output path:
```
../FactualityPrompt/outputs/cp/factual/qwen3_1.7b/offset2000_n2000.jsonl
```

---

## 📈 Step 2: Evaluation

Switch to the `FactualityPrompt` folder and run the evaluation script:

### 🧪 Basic Evaluation (non-CP)
```bash
cd FactualityPrompt

PYTHONPATH=. python src/evaluate_v3_final.py \
  --prompt_type nonfactual \
  --model_name qwen3_1.7b
```

### 🧪 Evaluation with CP-generated outputs
```bash
PYTHONPATH=. python src/evaluate_v3_final.py \
  --prompt_type factual \
  --model_name qwen3_0.6b \
  --cp_flag cp
```

> The script will automatically find and merge up to 4 files:
> `offset0000_n2000.jsonl`, `offset2000_n2000.jsonl`, ...

---

## 🔍 Evaluation Options

| Argument        | Description                                      |
|-----------------|--------------------------------------------------|
| `--prompt_type` | `factual` or `nonfactual`                        |
| `--model_name`  | e.g., `qwen3_1.7b`                                |
| `--cp_flag`     | Optional. `cp` or `noncp` (default is `noncp`)   |

---

## ✅ Requirements

Install required packages in both components:

```bash
# For generation
cd factuality_enhanced_lm_hf
pip install -r requirements.txt

# For evaluation
cd ../FactualityPrompt
pip install -r requirements.txt
```

---

## 💡 Notes

- You can generate with or without conformal prediction (`cp_generate_qwen_jsonl.py` vs `generate_qwen_jsonl.py`)
- Make sure outputs follow the expected directory and naming format
- For large-scale jobs, the provided `qwen_gen.sh` uses GPU round-robin scheduling

---

## 📄 Citation

If you use this pipeline or evaluation protocol, please cite <Sorry, research is ongoing!>
