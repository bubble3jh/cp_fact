# qwen 0.6B
CUDA_VISIBLE_DEVICES=0 python cp_generate_qwen_jsonl.py --fact_type factual --n_lines 2000 --offset 0 &
CUDA_VISIBLE_DEVICES=1 python cp_generate_qwen_jsonl.py --fact_type factual --n_lines 2000 --offset 2000 &
CUDA_VISIBLE_DEVICES=2 python cp_generate_qwen_jsonl.py --fact_type factual --n_lines 2000 --offset 4000 &
CUDA_VISIBLE_DEVICES=3 python cp_generate_qwen_jsonl.py --fact_type factual --n_lines 2000 --offset 6000 &
wait

CUDA_VISIBLE_DEVICES=0 python cp_generate_qwen_jsonl.py --fact_type nonfactual --n_lines 2000 --offset 0 &
CUDA_VISIBLE_DEVICES=1 python cp_generate_qwen_jsonl.py --fact_type nonfactual --n_lines 2000 --offset 2000 &
CUDA_VISIBLE_DEVICES=2 python cp_generate_qwen_jsonl.py --fact_type nonfactual --n_lines 2000 --offset 4000 &
CUDA_VISIBLE_DEVICES=3 python cp_generate_qwen_jsonl.py --fact_type nonfactual --n_lines 2000 --offset 6000 &
wait

# qwen 1.7B
CUDA_VISIBLE_DEVICES=0 python cp_generate_qwen_jsonl.py --model_name Qwen/Qwen3-1.7B --fact_type factual --n_lines 2000 --offset 0 &
CUDA_VISIBLE_DEVICES=1 python cp_generate_qwen_jsonl.py --model_name Qwen/Qwen3-1.7B --fact_type factual --n_lines 2000 --offset 2000 &
CUDA_VISIBLE_DEVICES=2 python cp_generate_qwen_jsonl.py --model_name Qwen/Qwen3-1.7B --fact_type factual --n_lines 2000 --offset 4000 &
CUDA_VISIBLE_DEVICES=3 python cp_generate_qwen_jsonl.py --model_name Qwen/Qwen3-1.7B --fact_type factual --n_lines 2000 --offset 6000 &
wait

CUDA_VISIBLE_DEVICES=0 python cp_generate_qwen_jsonl.py --model_name Qwen/Qwen3-1.7B --fact_type nonfactual --n_lines 2000 --offset 0 &
CUDA_VISIBLE_DEVICES=1 python cp_generate_qwen_jsonl.py --model_name Qwen/Qwen3-1.7B --fact_type nonfactual --n_lines 2000 --offset 2000 &
CUDA_VISIBLE_DEVICES=2 python cp_generate_qwen_jsonl.py --model_name Qwen/Qwen3-1.7B --fact_type nonfactual --n_lines 2000 --offset 4000 &
CUDA_VISIBLE_DEVICES=3 python cp_generate_qwen_jsonl.py --model_name Qwen/Qwen3-1.7B --fact_type nonfactual --n_lines 2000 --offset 6000 &
wait