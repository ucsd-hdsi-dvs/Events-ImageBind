#!/usr/bin/env bash

CHECK_PATH='/tsukimi/datasets/Chiba/imagebind_baseline/checkpoint'
LOG_DIR='/tsukimi/datasets/Chiba/imagebind_baseline/logs'

python train_baseline.py --full_model_checkpoint_dir ${CHECK_PATH}  \
    --full_model_checkpointing \
    --loggers tensorboard wandb \
    --loggers_dir ${CHECK_PATH} \
    --max_epochs 100 \
    --batch_size 8 \
    --lr 1e-3 \
    --prefix cut17_lr1e3_nolstm \
    --device cuda:4 \
    # --load_checkpoint '/tsukimi/datasets/Chiba/imagebind_baseline/checkpoint/last-v4.ckpt' \