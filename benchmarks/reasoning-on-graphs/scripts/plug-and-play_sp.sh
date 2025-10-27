# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
# Copyright (c) 2023 Linhao Luo (Raymond)


SPLIT="test"
DATASET_LIST="test_def_scores.pkl" # output of compute_neighs_and_sp.py
BEAM_LIST="5" # "1 2 3 4 5"
MODEL_LIST="gpt-4o-mini"
PROMPT_LIST="prompts/general_prompt.txt"
set -- $PROMPT_LIST

for i in $(seq 1 3); do
    for DATA_NAME in $DATASET_LIST; do
        for N_BEAM in $BEAM_LIST; do
            RULE_PATH=results/gen_rule_path/${DATA_NAME}/RoG-sp$i/test/predictions_${N_BEAM}_False.jsonl
            for k in "${!MODEL_LIST[@]}"; do
            
                MODEL_NAME=${MODEL_LIST[$k]}
                PROMPT_PATH=${PROMPT_LIST[$k]}
                
                python src/qa_prediction/predict_answer.py \
                    --model_name ${MODEL_NAME} \
                    -d ${DATA_NAME} \
                    --prompt_path ${PROMPT_PATH} \
                    --add_rule \
                    --rule_path ${RULE_PATH}
            done
        done
    done
done