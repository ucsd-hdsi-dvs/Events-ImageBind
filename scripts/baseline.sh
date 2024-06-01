#!/usr/bin/env bash

CHECK_PATH='/tsukimi/datasets/Chiba/imagebind_baseline/checkpoint'
LOG_DIR='/tsukimi/datasets/Chiba/imagebind_baseline/logs'

python train_baseline.py --full_model_checkpoint_dir ${CHECK_PATH}  \
    --full_model_checkpointing \
    --loggers tensorboard \
    --loggers_dir ${CHECK_PATH} \
    --max_epochs 100 \
    --batch_size 2 \
    --lr 1e-4 \
    --prefix firstrun \
    --device cuda:4 \