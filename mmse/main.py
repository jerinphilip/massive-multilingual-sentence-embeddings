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
from mmse.utils.logging import prettified
from fairseq.meters import AverageMeter

def update_state(epoch, loss_meter, loss, batch, state_dict):
    B, T = batch["batch_size"], batch["tgt_num_tokens"]
    loss_meter.update(loss)
    state_dict.update({
        "epoch": epoch,
        "loss": loss_meter.avg,
        "ntokens": batch["src_num_tokens"] + batch["tgt_num_tokens"],
    })


def validate(args, epoch, trainer, loader, state_dict):
    loss = AverageMeter()
    # progress = progress_handler.get(args.progress)
    progress = progress_handler.get('none')
    pbar = progress(enumerate(loader), state_dict, total=len(loader), desc='dev')
    for batch_idx, batch in pbar:
        mini_batch_loss = trainer.valid_step(batch)
        update_state(epoch, loss, mini_batch_loss, batch, state_dict)
        # trainer.debug(batch)
    print('validation epoch', epoch, prettified(state_dict.items()))

    # trainer._lr_scheduler.step(loss.avg)

def train(args, epoch, trainer, loader, state_dict):
    loss = AverageMeter()
    progress = progress_handler.get(args.progress)
    pbar = progress(enumerate(loader), state_dict, total=len(loader), desc='train')
    for batch_idx, batch in pbar:
        mini_batch_loss = trainer.train_step(batch)
        update_state(epoch, loss, mini_batch_loss, batch, state_dict)
        if (batch_idx + 1)% args.update_every == 0:
            Checkpoint.save(args, trainer, state_dict)
        trainer.debug(batch)
    print('train epoch', epoch, prettified(state_dict))

def run_epoch(args, epoch, trainer, loader, state_dict):
    loss = AverageMeter()
    progress = progress_handler.get(args.progress)
    pbar = progress(enumerate(loader["train"]), state_dict, total=len(loader['train']), desc='train')
    for batch_idx, batch in pbar:
        mini_batch_loss = trainer.train_step(batch)
        update_state(epoch, loss, mini_batch_loss, batch, state_dict)
        if (batch_idx + 1)% args.update_every == 0:
            Checkpoint.save(args, trainer, state_dict)

        if (batch_idx + 1)% args.validate_every == 0:
            validate(args, epoch, trainer, loader["dev"], state_dict)

        # trainer.debug(batch)
    print('train epoch', epoch, prettified(state_dict.items()))

def main(args, init_distributed=True):
    task = JointSpaceLearningTask(args)
    task.setup_task()
    loader = task.get_loader()

    if init_distributed:
        args.distributed_rank = distributed.distributed_init(args)

    model = EmbeddingModel.build(args, task.dictionary)
    trainer = Trainer(args, model)
    state_dict = {}
    Checkpoint.load(args, trainer, state_dict)
    resume_epoch = state_dict.get('epoch', 0)

    for epoch in range(resume_epoch, args.num_epochs):
        run_epoch(args, epoch, trainer, loader, state_dict)

def distributed_main(i, args, start_rank=0):
    args.device = i
    if args.distributed_rank is None:
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    port = args.distributed_port
    host = args.distributed_master_addr
    args.distributed_init_method = 'tcp://{host}:{port}'.format(host=host, port=port)
    args.distributed_rank = None  # set based on device id
    #if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
    #    print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
    assert args.distributed_world_size <= torch.cuda.device_count()
    torch.multiprocessing.spawn(
        fn=distributed_main,
        args=(args, ),
        nprocs=args.distributed_world_size,
    )

    # main(args, init_distributed=False)

