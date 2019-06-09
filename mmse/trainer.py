from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel
from .utils.device_utils import move_to
import torch
from mmse.utils.distributed import all_gather_list
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Trainer:
    def __init__(self, args, model):
        self.args = args
        self._model = model
        self._reset_state()
    
    def _reset_state(self):
        self._model = self._model.to(self.device)
        self._wrapped_model = None
        self._optimizer = torch.optim.Adam(self._model.parameters())
        self._lr_scheduler = ReduceLROnPlateau(self._optimizer, mode='min', patience=3)

    @property
    def device(self):
        if self.args.distributed_rank is None:
            return torch.device("cuda")
        else:
            device_name = "cuda:{}".format(self.args.device)
            return torch.device(device_name)

    @property
    def model(self):
        args = self.args
        if self.args.distributed_rank is None:
            return self._model
        else:
            if self._wrapped_model is None:
                self._wrapped_model = DistributedDataParallel(
                    module=self._model,
                    device_ids=[args.device],
                    output_device=args.device,
                    # broadcast_buffers=False
                )
            return self._wrapped_model

    def train_step(self, sample):
        args = self.args
        self.model.train()
        self._optimizer.zero_grad()
        sample = move_to(sample, self.device)
        loss, logging_outputs = self.model(sample)
        loss.backward()
        clip_grad_norm_(self._model.parameters(), args.max_grad_norm)
        self._optimizer.step()
        return loss.item()

    def valid_step(self, sample):
        self.model.eval()
        with torch.no_grad():
            loss, logging_outputs = self.model(sample)
            return loss.item()


    def debug(self, batch):
        gout = self.model.get_generator_output(batch)
        _max, argmax = torch.max(gout, dim=2)
        argmax = argmax.transpose(0, 1).contiguous()
        print(argmax) 
        print(batch["tgts"][:, 1:])

    def state_dict(self):
        _export = {
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "lr_schedular": self._lr_scheduler.state_dict()
        }
        move_to(_export, torch.device("cpu"))
        return _export

    def load_state_dict(self, state_dict):
        self._model.load_state_dict(state_dict['model'])
        self._reset_state()
        self._optimizer.load_state_dict(state_dict['optimizer'])
        if 'lr_scheduler' in state_dict:
            self._lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
