from torch.nn.utils.clip_grad import clip_grad_norm_

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



