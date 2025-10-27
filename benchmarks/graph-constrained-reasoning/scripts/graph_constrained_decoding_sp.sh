# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
# Copyright (c) 2024 Linhao Luo (Raymond)


DATA_PATH=../../GTSQA
DATA_LIST=test_def_scores.pkl
SPLIT="test"
INDEX_LEN=2
ATTN_IMP=flash_attention_2


K="10" # 3 5 10 20
for i in $(seq 1 3); do
  for DATA in ${DATA_LIST}; do
    for k in $K; do
      python workflow/predict_paths_and_answers.py --kg_path ../../data/ogbl_wikikg2 --data_path ${DATA_PATH} --d ${DATA} --split ${SPLIT} --index_path_length ${INDEX_LEN} --model_name GCR-sp$i --model_path save_models/def/GCR-Meta-Llama-3.1-8B-Instruct-sp$i --k ${k} --prompt_mode zero-shot --generation_mode group-beam --attn_implementation ${ATTN_IMP}
    done
  done
done