#!bin/bash

module load python/3.7.0
module load pytorch/1.0.0
module load pytorch/fairseq/0.6.2

IMPORTS=(
    filtered-iitb.tar
    ilci.tar
    national-newscrawl.tar
    ufal-en-tam.tar
    wat-ilmpc.tar
    LASER-tatoeba.tar
)

LOCAL_ROOT="/ssd_scratch/cvit/$USER"
REMOTE_ROOT="ada:/share1/dataset/text"


mkdir -p $LOCAL_ROOT/{data,checkpoints}

DATA=$LOCAL_ROOT/data
CHECKPOINTS=$LOCAL_ROOT/checkpoints

function copy {
    for IMPORT in ${IMPORTS[@]}; do
        rsync --progress $REMOTE_ROOT/$IMPORT $DATA/
        tar_args="$DATA/$IMPORT -C $DATA/"
        tar -df $tar_args 2> /dev/null || tar -kxvf $tar_args
    done
}

copy
export ILMULTI_CORPUS_ROOT=$DATA
HOSTNAME=$(hostname)

python3 -m mmse.main                    \
    --lr 1e-5                           \
    --config_file configs/multi.yaml    \
    --max_tokens 2000                   \
    --num_epochs 200                    \
    --distributed_backend nccl          \
    --distributed_master_addr $HOSTNAME \
    --distributed_world_size 4          \
    --distributed_port 1947             \
    --progress tqdm                     \
    --update_every 1000                 \
    --save_path $CHECKPOINTS/checkpoint.pt
    # --dict_path data/dicts/central.dict \
    # --prefix $DATA/ufal-en-tam/ \
    # --config_file configs/clean.yaml    \
    # --source_lang en \
    # --target_lang ta \
