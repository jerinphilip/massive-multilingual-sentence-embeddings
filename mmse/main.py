import os
import sys
import torch
from mmse.parser import create_parser
from mmse.task import JointSpaceLearningTask
from mmse.models import EmbeddingModel
from mmse.trainer import Trainer
from mmse.utils import distributed
from mmse.utils.progress import progress_handler
from mmse.utils.checkpoint import Checkpoint

def train(args, trainer, loader, state_dict):
    loss_sum = 0
    progress = progress_handler.get(args.progress)
    state_dict = {}
    pbar = progress(enumerate(loader), state_dict, total=len(loader))
    for batch_idx, batch in pbar:
        loss = trainer.train_step(batch)
        loss_sum += loss
        if batch_idx % args.update_every == 0:
            state_dict.update({
                "lpb": loss_sum/(batch_idx+1),
                "lpt": loss_sum/((batch_idx+1)*batch["tgt_num_tokens"]),
                "toks": batch["src_num_tokens"] + batch["tgt_num_tokens"],
            })

            Checkpoint.save(args, trainer, state_dict)



def main(args, init_distributed=True):
    task = JointSpaceLearningTask(args)
    task.setup_task()
    loaders = task.get_loader()
    torch.cuda.set_device(args.device)

    if init_distributed:
        args.distributed_rank = distributed.distributed_init(args)

    model = EmbeddingModel.build(args, task.dictionary)
    trainer = Trainer(args, model)
    state_dict = {}
    Checkpoint.load(args, trainer, state_dict)
    loader = loaders[0]
    for epoch in range(args.num_epochs):
        train(args, trainer, loader, state_dict)

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

