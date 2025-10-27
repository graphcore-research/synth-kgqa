# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
# Copyright (c) 2023 Linhao Luo (Raymond)


MODEL_PATH=meta-llama/Llama-2-7b-chat-hf
DATASET_LIST="datasets/joint_training/align/synthetic_def/synthetic_def_train.jsonl"
SAVE_NAME=RoG
SAVE_PATH=save_models/${SAVE_NAME}
ADD_REL=False

for i in $(seq 1 3); do
    accelerate launch --config_file config/deepspeed_zero3.yml src/joint_training/joint_finetuning.py \
        --data_path_list ${DATASET_LIST}  \
        --model_name_or_path ${MODEL_PATH} \
        --output_dir save_models/${SAVE_NAME}-$i \
        --add_rel_token ${ADD_REL} \
        --bf16 True \
        --num_train_epochs 4 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 16 \
        --evaluation_strategy "no" \
        --save_strategy "no" \
        --save_steps 500 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --report_to "wandb" \
        --gradient_checkpointing True \
        --run_name ${SAVE_NAME}-$i
done