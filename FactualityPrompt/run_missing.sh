#!/bin/bash

# 기본 경로 (필요 시 수정)
BASE_DIR="/data3/bubble3jh/FactualityPrompt/outputs"
PYTHON_SCRIPT="src/evaluate_v3_final.py"

# 유형별 루프
for MODE in cp noncp; do
  for FACT_TYPE in factual nonfactual; do
    for MODEL in qwen3_0.6b qwen3_1.7b; do

      MERGED_PATH="${BASE_DIR}/${MODE}/${FACT_TYPE}/${MODEL}/merged.jsonl"
      RESULT_PATH="${MERGED_PATH/_merged.jsonl/_results.jsonl}"

      if [ ! -f "$RESULT_PATH" ]; then
        echo "[RUNNING] $MODE / $FACT_TYPE / $MODEL"

        CMD="PYTHONPATH=. python $PYTHON_SCRIPT --prompt_type $FACT_TYPE --model_name $MODEL"
        if [ "$MODE" = "cp" ]; then
          CMD="$CMD --cp"
        fi

        echo "[COMMAND] $CMD"
        eval $CMD
        echo "-------------------------------------------------"
      else
        echo "[SKIP] Already exists → $RESULT_PATH"
      fi

    done
  done
done
