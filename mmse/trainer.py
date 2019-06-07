from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel
from .utils.device_utils import move_to
import torch
from .distributed_utils import all_gather_list

class Trainer:
    def __init__(self, args, model):
        self.args = args
        # Detect device
        self._model = model
        self.build_optimizer()
        self._model = self._model.to(self.device)
        self._wrapped_model = None
    
    def build_optimizer(self):
        self._optimizer = torch.optim.Adam(self._model.parameters())

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
        # gathered = all_gather_list([loss, logging_outputs])
        # print(gathered)

        loss.backward()
        # All gather.
        # Multiply gradients * world_size / sample_size
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

