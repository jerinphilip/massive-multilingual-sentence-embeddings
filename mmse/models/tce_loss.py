import torch
from torch import nn

class TCELoss(nn.Module):
    def __init__(self, dictionary):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='sum', 
                ignore_index=dictionary.pad())

    def forward(self, logits, classes):
        T, B, H = logits.size()
        _B, _T = classes.size()

        # print(logits.size(), classes.size())

        # T x B x H -> B x T x H
        logits = logits.transpose(0, 1)
        logits = logits.contiguous().view(T*B, H)
        classes = classes.contiguous().view(-1)
        return self.criterion(logits, classes)


