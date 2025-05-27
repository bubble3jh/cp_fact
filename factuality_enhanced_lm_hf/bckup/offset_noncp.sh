# 8000개를 2000개씩 병렬 생성
CUDA_VISIBLE_DEVICES=4 python generate_qwen_jsonl.py --fact_type factual --n_lines 2000 --offset 0 &
CUDA_VISIBLE_DEVICES=5 python generate_qwen_jsonl.py --fact_type factual --n_lines 2000 --offset 2000 &
CUDA_VISIBLE_DEVICES=6 python generate_qwen_jsonl.py --fact_type factual --n_lines 2000 --offset 4000 &
CUDA_VISIBLE_DEVICES=7 python generate_qwen_jsonl.py --fact_type factual --n_lines 2000 --offset 6000 &
wait

CUDA_VISIBLE_DEVICES=4 python generate_qwen_jsonl.py --fact_type nonfactual --n_lines 2000 --offset 0 &
CUDA_VISIBLE_DEVICES=5 python generate_qwen_jsonl.py --fact_type nonfactual --n_lines 2000 --offset 2000 &
CUDA_VISIBLE_DEVICES=6 python generate_qwen_jsonl.py --fact_type nonfactual --n_lines 2000 --offset 4000 &
CUDA_VISIBLE_DEVICES=7 python generate_qwen_jsonl.py --fact_type nonfactual --n_lines 2000 --offset 6000 &
wait

# qwen 1.7B
CUDA_VISIBLE_DEVICES=4 python generate_qwen_jsonl.py --model_name Qwen/Qwen3-1.7B --fact_type factual --n_lines 2000 --offset 0 &
CUDA_VISIBLE_DEVICES=5 python generate_qwen_jsonl.py --model_name Qwen/Qwen3-1.7B --fact_type factual --n_lines 2000 --offset 2000 &
CUDA_VISIBLE_DEVICES=6 python generate_qwen_jsonl.py --model_name Qwen/Qwen3-1.7B --fact_type factual --n_lines 2000 --offset 4000 &
CUDA_VISIBLE_DEVICES=7 python generate_qwen_jsonl.py --model_name Qwen/Qwen3-1.7B --fact_type factual --n_lines 2000 --offset 6000 &
wait

CUDA_VISIBLE_DEVICES=4 python generate_qwen_jsonl.py --model_name Qwen/Qwen3-1.7B --fact_type nonfactual --n_lines 2000 --offset 0 &
CUDA_VISIBLE_DEVICES=5 python generate_qwen_jsonl.py --model_name Qwen/Qwen3-1.7B --fact_type nonfactual --n_lines 2000 --offset 2000 &
CUDA_VISIBLE_DEVICES=6 python generate_qwen_jsonl.py --model_name Qwen/Qwen3-1.7B --fact_type nonfactual --n_lines 2000 --offset 4000 &
CUDA_VISIBLE_DEVICES=7 python generate_qwen_jsonl.py --model_name Qwen/Qwen3-1.7B --fact_type nonfactual --n_lines 2000 --offset 6000 &
wait