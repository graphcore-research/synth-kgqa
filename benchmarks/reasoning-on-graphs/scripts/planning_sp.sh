# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
# Copyright (c) 2023 Linhao Luo (Raymond)


SPLIT="test"
DATASET_LIST="test_def_scores.pkl" # output of compute_neighs_and_sp.py
MODEL_NAME=RoG

BEAM_LIST="5" # "1 2 3 4 5"
for i in $(seq 1 3); do
    for DATASET in $DATASET_LIST; do
        for N_BEAM in $BEAM_LIST; do
            python src/qa_prediction/gen_rule_path.py \
            --model_name RoG-sp$i \
            --model_path save_models/RoG-sp$i \
            -d ${DATASET} \
            --split ${SPLIT} \
            --n_beam ${N_BEAM}
        done
    done
done