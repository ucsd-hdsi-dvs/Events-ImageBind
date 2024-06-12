#!/usr/bin/env bash

data_dir='/tsukimi/datasets/Chiba/finetune_train'
check_dir='/tsukimi/datasets/Chiba/finetune_checkpoint/checkpoints'
log_dir='/tsukimi/datasets/Chiba/finetune_checkpoint/logs'

python train_events.py --full_model_checkpoint_dir ${check_dir} \
    --full_model_checkpointing \
    --device cuda \
    --datasets_dir ${data_dir} \
    --loggers wandb tensorboard \
    --loggers_dir ${log_dir} \
    --max_epochs 100 \
    --headless \
    --num_workers 4 \
    # --lora \

