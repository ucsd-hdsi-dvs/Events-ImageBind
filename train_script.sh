#!/usr/bin/env bash

data_dir='/tsukimi/datasets/MVSEC/event_chunks_imgbind'
check_dir='/tsukimi/datasets/Chiba/finetune_checkpoint/checkpoints'
log_dir='/tsukimi/datasets/Chiba/finetune_checkpoint/logs'
checkpoint_path='/tsukimi/datasets/Chiba/finetune_checkpoint/checkpoints/imagebind-epoch=12-val_loss=0.30.ckpt'

python train_events.py --full_model_checkpoint_dir ${check_dir} \
    --full_model_checkpointing \
    --device cuda \
    --datasets_dir ${data_dir} \
    --loggers wandb \
    --loggers_dir ${log_dir} \
    --max_epochs 100 \
    --headless \
    --num_workers 4 \
    --batch_size 8 \
    # --checkpoint_path ${checkpoint_path} \


