#!/bin/bash

# Configuration
data_root=""
resolution="512"
dataloader_num_workers="8"
train_batch_size="16"
gradient_accumulation_steps="1"
pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"
pred_type="epsilon"
loss_type="noise_matching"
windows="4"
solving_steps="8"
learning_rate="8e-5"
lr_scheduler="constant"
lr_warmup_steps="1000"
adam_weight_decay="1e-5"
max_grad_norm="1"
max_train_steps="1000000"
use_ema=""
mixed_precision="fp16"
output_dir="_exps_/tmp"
debug=""

# Run training
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --main_process_port 10764 \
    --num_processes 1 \
    --num_cpu_threads_per_process 8 \
    --mixed_precision "fp16" \
    scripts/train_perflow_debug.py \
    --resolution ${resolution} \
    --dataloader_num_workers ${dataloader_num_workers} \
    --train_batch_size ${train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --pretrained_model_name_or_path ${pretrained_model_name_or_path} \
    --pred_type ${pred_type} \
    --loss_type ${loss_type} \
    --windows ${windows} \
    --solving_steps ${solving_steps} \
    --support_cfg \
    --cfg_sync \
    --learning_rate ${learning_rate} \
    --lr_scheduler ${lr_scheduler} \
    --lr_warmup_steps ${lr_warmup_steps} \
    --adam_weight_decay ${adam_weight_decay} \
    --max_grad_norm ${max_grad_norm} \
    --max_train_steps ${max_train_steps} \
    --use_ema \
    --mixed_precision ${mixed_precision} \
    --output_dir ${output_dir} \
    --debug
# Cleanup if necessary
