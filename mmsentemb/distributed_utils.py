"""
Taken from
https://github.com/pytorch/fairseq/blob/cf17068aad5af49eb9a106e314f617b15438dc9e/fairseq/distributed_utils.py

Licenses of the repo apply.

"""
from collections import namedtuple
import os
import pickle
import socket
import subprocess
import warnings

import torch
import torch.distributed as dist
from torch import nn

def is_master(args):
    return args.distributed_rank == 0

def infer_init_method(args):
    if args.distributed_init_method is not None:
        return

    # support torch.distributed.launch
    if all(key in os.environ for key in [
        'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK'
    ]):
        args.distributed_init_method = 'env://'
        args.distributed_world_size = int(os.environ['WORLD_SIZE'])
        args.distributed_rank = int(os.environ['RANK'])

    # we can determine the init method automatically for Slurm
    elif args.distributed_port > 0:
        node_list = os.environ.get('SLURM_JOB_NODELIST')
        if node_list is not None:
            try:
                hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', node_list])
                args.distributed_init_method = 'tcp://{host}:{port}'.format(
                    host=hostnames.split()[0].decode('utf-8'),
                    port=args.distributed_port,
                )
                nnodes = int(os.environ.get('SLURM_NNODES'))
                ntasks_per_node = int(os.environ.get('SLURM_NTASKS_PER_NODE'))
                if ntasks_per_node == 1:
                    assert args.distributed_world_size % nnodes == 0
                    gpus_per_node = args.distributed_world_size // nnodes
                    node_id = int(os.environ.get('SLURM_NODEID'))
                    args.distributed_rank = node_id * gpus_per_node
                else:
                    assert ntasks_per_node == args.distributed_world_size // nnodes
                    args.distributed_no_spawn = True
                    args.distributed_rank = int(os.environ.get('SLURM_PROCID'))
                    args.device_id = int(os.environ.get('SLURM_LOCALID'))
            except subprocess.CalledProcessError as e:  # scontrol failed
                raise e
            except FileNotFoundError:  # Slurm is not installed
                pass

def distributed_init(args):
    if args.distributed_world_size == 1:
        raise ValueError('Cannot initialize distributed with distributed_world_size=1')

    if torch.distributed.is_initialized():
        warnings.warn('Distributed is already initialized, cannot initialize twice!')
    else:
        print('| distributed init (rank {}): {}'.format(
            args.distributed_rank, args.distributed_init_method), flush=True)
        dist.init_process_group(
            backend=args.distributed_backend,
            init_method=args.distributed_init_method,
            world_size=args.distributed_world_size,
            rank=args.distributed_rank,
        )
        print('| initialized host {} as rank {}'.format(
            socket.gethostname(), args.distributed_rank), flush=True)

        # perform a dummy all-reduce to initialize the NCCL communicator
        dist.all_reduce(torch.rand(1).cuda())

        suppress_output(is_master(args))

    args.distributed_rank = torch.distributed.get_rank()
    return args.distributed_rank

def suppress_output(is_master):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def get_default_group():
    return dist.group.WORLD


def all_reduce(tensor, group=None):
    if group is None:
        group = get_default_group()
    return dist.all_reduce(tensor, group=group)

