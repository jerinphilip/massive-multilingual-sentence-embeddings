import os
import sys
import torch
from .parser import create_parser
from .task import JointSpaceLearningTask
from .models import EmbeddingModel
from .trainer import Trainer
from . import distributed_utils
from .utils.progress import progress_handler

def train(args, trainer, task, loaders):
    loss_sum = 0
    loaders = loaders[:1]
    progress = progress_handler.get(args.progress)
    for dataset_idx, loader in enumerate(loaders):
        state_dict = {}
        # iterator = iter(loader)
        pbar = progress(enumerate(loader), state_dict, 
                total=len(loader), ascii='#', leave=True)
        for batch_idx, batch in pbar:
            loss = trainer.train_step(batch)
            loss_sum += loss
            if batch_idx % args.update_every == 0:
                # print(loss/batch['seq_length'])
                state_dict.update({
                    # "epoch": epoch,
                    #"batch_idx": batch_idx,
                    "dataset_idx": dataset_idx,
                    "lpb": loss_sum/(batch_idx+1),
                    "lpt": loss_sum/((batch_idx+1)*batch["tgt_num_tokens"]),
                    "toks": batch["src_num_tokens"] + batch["tgt_num_tokens"],
                    # "src_toks": batch["src_num_tokens"],
                    # "tgt_toks": batch["tgt_num_tokens"]
                })


def main(args, init_distributed=True):
    task = JointSpaceLearningTask(args)
    task.setup_task()
    loaders = task.get_loader()
    torch.cuda.set_device(args.device)

    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    model = EmbeddingModel.build(args, task.dictionary)
    trainer = Trainer(args, model)
    for epoch in range(args.num_epochs):
        train(args, trainer, task, loaders)

def distributed_main(i, args, start_rank=0):
    args.device = i
    if args.distributed_rank is None:
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    assert args.distributed_world_size <= torch.cuda.device_count()
    port = args.distributed_port
    host = args.distributed_master_addr
    args.distributed_init_method = 'tcp://{host}:{port}'.format(host=host, port=port)
    args.distributed_rank = None  # set based on device id
    #if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
    #    print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
    torch.multiprocessing.spawn(
        fn=distributed_main,
        args=(args, ),
        nprocs=args.distributed_world_size,
    )

    # main(args, init_distributed=False)

