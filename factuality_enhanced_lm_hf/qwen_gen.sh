#!/bin/bash

MODELS=("Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B")
FACT_TYPES=("factual" "nonfactual")
CP_FLAGS=("cp" "noncp")
OFFSETS=(0 2000 4000 6000)
GPUS=(0 1 2 3 4 5 6 7)

i=0  # job counter for GPU round-robin

for model in "${MODELS[@]}"; do
  for fact in "${FACT_TYPES[@]}"; do
    for cp_flag in "${CP_FLAGS[@]}"; do
      for offset in "${OFFSETS[@]}"; do
        gpu_id=${GPUS[$((i % ${#GPUS[@]}))]}
        if [ "$cp_flag" = "cp" ]; then
          script="cp_generate_qwen_jsonl.py"
        else
          script="generate_qwen_jsonl.py"
        fi

        echo "Launching: $script on GPU $gpu_id (model=$model, fact=$fact, offset=$offset)"
        CUDA_VISIBLE_DEVICES=$gpu_id python $script \
          --model_name "$model" \
          --fact_type "$fact" \
          --n_lines 2000 \
          --offset $offset &

        ((i+=1))

        # Every 8 jobs, wait to prevent overload
        if (( i % 8 == 0 )); then
          wait
        fi
      done
    done
  done
done

# Final wait to catch remaining background jobs
wait
