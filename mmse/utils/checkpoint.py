import os
import torch
import shutil
import warnings
from mmse.utils import distributed
from mmse.utils.device_utils import move_to
from threading import Thread
import time

class Checkpoint:
    @staticmethod
    def save(args, trainer, train_state_dict):
        if not distributed.is_master(args):
            return

        state_dict = {}

        # Load trainer
        state_dict.update(trainer.state_dict())
        state_dict.update({"update_itr": train_state_dict})
        state_dict.update({"args": vars(args)})

        move_to(state_dict, torch.device("cpu"))
        backup = '{}.old'.format(args.save_path)
        def write_closure():
            if os.path.exists(args.save_path):
                shutil.copyfile(args.save_path, backup)
            torch.save(state_dict, args.save_path)

        # https://stackoverflow.com/a/842567/4565794
        # How to prevent a block of code from being interrupted by
        # KeyboardInterrupt in Python?
        a = Thread(target=write_closure)
        start = time.time()
        a.start()
        a.join()
        end = time.time()
        duration = end - start
        print("Checkpoint saved. Writing took {:.2f} seconds".format(duration))

    @staticmethod
    def load(args, trainer, train_state_dict):
        if os.path.exists(args.save_path):
            checkpoint = torch.load(args.save_path, map_location=torch.device("cpu"))
            # Update trainer
            trainer_state_dict = trainer.state_dict()
            trainer_state_dict.update(checkpoint)
            trainer.load_state_dict(trainer_state_dict)

            # del checkpoint['model']
            # del checkpoint['optimizer']
            # Update epoch
            train_state_dict.update(checkpoint["update_itr"])
            # train_state_dict.update(checkpoint)
        else:
            warnings.warn("Checkpoint not found, training from scratch")


