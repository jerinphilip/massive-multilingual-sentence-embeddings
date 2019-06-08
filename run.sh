#!/bin/bash

module load use.own
module load python/3.7.0
module load pytorch/1.0.0
module load nccl/2.2.13

# HOSTNAME=$(hostname)
ROOT='/ssd_scratch/cvit/perf-test'
HOSTNAME='localhost'
python3 -m mmse.main \
    --source $ROOT/data/cricket/train.en \
    --source_lang en \
    --target $ROOT/data/cricket/train.hi \
    --target_lang hi \
    --dict_path data/dicts/central.dict \
    --max_tokens 1800 \
    --num_epochs 200 \
    --distributed_backend nccl \
    --distributed_world_size 4 \
    --distributed_master_addr $HOSTNAME \
    --distributed_port 1947 \
    --progress tqdm \
    --save_path $ROOT/checkpoints/checkpoint.pt

