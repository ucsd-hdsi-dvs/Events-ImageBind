#!/usr/bin/env bash

data_dir='/tsukimi/datasets/N-Caltech101'

python train_caltech.py --device cuda:4 \
    --datasets_dir ${data_dir} \
    --loggers wandb \
    --max_epochs 100 \
    --num_workers 4 \
    --batch_size 8 \