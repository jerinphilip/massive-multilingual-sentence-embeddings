from torch.nn.utils.clip_grad import clip_grad_norm_
import torch

class Trainer:
    def __init__(self, args, model, optimizer, logger):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.logger = logger

    def to(self, device):
        self.model.to(device)

    def run_update(self, batch):
        self.optimizer.zero_grad()
        loss = self.model(batch)
        loss = loss/batch["batch_size"]
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
        self.optimizer.step()
        return loss.item()


    def debug(self, batch):
        gout = self.model.get_generator_output(batch)
        _max, argmax = torch.max(gout, dim=2)
        argmax = argmax.transpose(0, 1).contiguous()
        print(argmax) 
        print(batch["tgts"][:, 1:])


