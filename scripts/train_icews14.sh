#!/usr/bin/env bash

set -x
set -e

TASK="ICEWS14"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model "./bert-base-uncased" \
--pooling mean \
--lr 5e-4 \
--use-timehistory-graph \
--train-path "$DATA_DIR/train.txt.json" \
--valid-path "$DATA_DIR/test.txt.json" \
--task ${TASK} \
--batch-size 768 \
--prefix_projection \
--prompt_hidden_dim 512 \
--max-num-tokens 60 \
--prompt_length 20 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--finetune-t \
--pre-batch 2 \
--epochs 15 \
--workers 4 \
--seed 4399 \
--max-to-keep 10 "$@"
