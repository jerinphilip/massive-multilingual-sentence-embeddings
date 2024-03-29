from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel
from .utils.device_utils import move_to
import torch
from mmse.utils.distributed import all_gather_list
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .utils.distributed import all_gather_list

class Trainer:
    def __init__(self, args, model):
        self.args = args
        self._model = model
        self._reset_state()
    
    def _reset_state(self):
        self._model = self._model.to(self.device)
        self._wrapped_model = None
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.args.lr)
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
        loss, sample_sizes, logging_outputs = self.model(sample)
        loss.backward()

        # All gather sample sizes
        sample_sizes, logging_outputs = zip(*all_gather_list([sample_sizes, logging_outputs]))
        sample_sizes = list(sample_sizes)
        logging_outputs = list(logging_outputs)
        grad_denominator = sum(sample_sizes)
        self.multiply_grad(args.distributed_world_size/float(grad_denominator))
        max_norm = clip_grad_norm_(self._model.parameters(), args.max_grad_norm)
        self._optimizer.step()
        torch.cuda.empty_cache()

        train_loss = loss.item()*args.distributed_world_size/float(grad_denominator)
        return train_loss

    def multiply_grad(self, val):
        """Multiplies grads by a constant *c*."""
        for p in self._model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(val)

    def valid_step(self, sample):
        self.model.eval()
        with torch.no_grad():
            sample = move_to(sample, self.device)
            loss, sample_sizes, logging_outputs = self.model(sample)
            sample_sizes, logging_outputs = zip(*all_gather_list([sample_sizes, logging_outputs]))
            sample_sizes = list(sample_sizes)
            logging_outputs = list(logging_outputs)

            grad_denominator = sum(sample_sizes)
            valid_loss = loss.item()*self.args.distributed_world_size/float(grad_denominator)
            return valid_loss


    def debug(self, sample):
        sample = move_to(sample, self.device)
        gout = self._model.get_generator_output(sample)
        _max, argmax = torch.max(gout, dim=2)
        argmax = argmax.transpose(0, 1).contiguous()
        print(argmax, flush=True) 
        print(sample["tgts"][:, 1:], flush=True)

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
