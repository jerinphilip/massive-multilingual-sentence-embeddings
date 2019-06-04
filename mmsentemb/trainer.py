
class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def run_update(self, batch):
        self.optimizer.zero_grad()
        loss = self.model(batch)
        loss = loss/batch["batch_size"]
        loss.backward()
        self.optimizer.step()
        return loss.item()



