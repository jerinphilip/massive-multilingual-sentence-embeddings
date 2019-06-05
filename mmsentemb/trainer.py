from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel
from .utils import move_to
import torch

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
        loss.backward()
        self._optimizer.step()
        return loss.item()


    def debug(self, batch):
        gout = self.model.get_generator_output(batch)
        _max, argmax = torch.max(gout, dim=2)
        argmax = argmax.transpose(0, 1).contiguous()
        print(argmax) 
        print(batch["tgts"][:, 1:])

