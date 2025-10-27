# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
# Copyright (c) 2024 Linhao Luo (Raymond)


DATA_PATH=../../GTSQA
DATA_LIST=test_def_scores.pkl
SPLIT="test"

MODEL_NAME=gpt-4o-mini
N_THREAD=10

for i in $(seq 1 3); do
  for DATA in ${DATA_LIST}; do
   REASONING_PATH="results/GenPaths/test_def_scores.pkl/GCR-$i/test/zero-shot-group-beam-k10-index_len2/predictions.jsonl"
    python workflow/predict_final_answer.py --kg-path ../../data/ogbl_wikikg2 --data_path ${DATA_PATH} --d ${DATA} --split ${SPLIT} --model_name ${MODEL_NAME} --reasoning_path ${REASONING_PATH} --add_path True -n ${N_THREAD}
  done
done
