#!/usr/bin/env bash

CHECK_PATH='/tsukimi/datasets/Chiba/imagebind_baseline/checkpoint'
LOG_DIR='/tsukimi/datasets/Chiba/imagebind_baseline/logs'

python train_baseline.py --full_model_checkpoint_dir ${CHECK_PATH}  \
    --full_model_checkpointing \
    --loggers tensorboard wandb \
    --loggers_dir ${CHECK_PATH} \
    --max_epochs 100 \
    --batch_size 1 \
    --lr 1e-3 \
    --prefix thirdrun_lr1e3_batch1 \
    --device cuda:4 \
    --load_checkpoint '/tsukimi/datasets/Chiba/imagebind_baseline/checkpoint/imagebind-epoch=07-val_acc=0.62.ckpt' \