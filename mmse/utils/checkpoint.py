import os
import torch
import warnings
from mmse.utils import distributed


class Checkpoint:
    @staticmethod
    def save(args, trainer, train_state_dict):
        if not distributed.is_master(args):
            return
        state_dict = {}
        state_dict.update(trainer.state_dict())
        state_dict.update(train_state_dict)
        torch.save(state_dict, args.save_path)


    @staticmethod
    def load(args, trainer, train_state_dict):
        if os.path.exists(args.save_path):
            checkpoint = torch.load(args.save_path)
            trainer_state_dict = trainer.state_dict()
            train_state_dict.update(checkpoint)
            trainer.load_state_dict(trainer_state_dict)
            train_state_dict.update(checkpoint)
        else:
            warnings.warn("Checkpoint not found, training from scratch")


