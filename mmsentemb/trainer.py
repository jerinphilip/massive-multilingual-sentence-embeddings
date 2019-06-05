from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel
from .utils import move_to
import torch

class Trainer:
    def __init__(self, args, model):
        self.args = args
        # Detect device
        self._model = model
        self._model = self._model.to(self.device)
        self._wrapped_model = None
        self.build_optimizer()
    
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
                    broadcast_buffers=False
                )
            return self._wrapped_model

    def run_update(self, batch):
        self.optimizer.zero_grad()
        loss = self.model(batch)
        loss = loss/batch["batch_size"]
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
        self.optimizer.step()
        return loss.item()

    def train_step(self, sample):
        args = self.args
        self.model.train()
        self._optimizer.zero_grad()
        sample = move_to(sample, self.device)
        # print(args.distributed_rank, "Model call")
        loss = self.model(sample)
        # print(args.distributed_rank, "Model call")
        # print(args.distributed_rank, loss.item())
        loss.backward()
        self._optimizer.step()
        return loss.item()


    def debug(self, batch):
        gout = self.model.get_generator_output(batch)
        _max, argmax = torch.max(gout, dim=2)
        argmax = argmax.transpose(0, 1).contiguous()
        print(argmax) 
        print(batch["tgts"][:, 1:])

    def build_trainer(self):
        self.model = EmbeddingModel.build(self.args, self.dictionary)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.logger = None
        self.trainer = Trainer(self.args, self.model, self.optimizer, self.logger)
