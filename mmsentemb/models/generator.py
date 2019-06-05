import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, args, dictionary):
        super().__init__()
        self.args = args
        self.dictionary = dictionary
        self.project = nn.Linear(args.decoder_output_size, len(dictionary))

    def forward(self, x):
        return self.project(x)

