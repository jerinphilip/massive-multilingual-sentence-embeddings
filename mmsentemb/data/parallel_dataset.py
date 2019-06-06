import os
from torch.utils.data import Dataset
import torch
import ilmulti as ilm
from collections import namedtuple
import random
from copy import deepcopy
from .corpus_flyweight import CORPUS_REGISTRY

def to_tensor(sample, lang_idx, eos_idx, shifted=False):
    idxs = deepcopy(sample)
    idxs.append(eos_idx)
    if shifted:
        idxs.insert(0, eos_idx)
    return torch.LongTensor(idxs), torch.LongTensor([lang_idx])

class ParallelDataset:
    def __init__(self, first, second, tokenizer, dictionary):
        self.tokenizer = tokenizer
        self.dictionary = dictionary
        self.first = CORPUS_REGISTRY.get(first, tokenizer, dictionary)
        self.second = CORPUS_REGISTRY.get(first, tokenizer, dictionary)

    def __len__(self):
        return len(self.first.data)

    def __getitem__(self, idx):
        idy = self.first.metadata.idxs[idx]
        eos_idx = self.dictionary.eos()

        def get(holding, idy, shifted):
            sample = holding.data[idy]
            lang = ilm.utils.language_token(holding.corpus.lang)
            lang_idx = self.dictionary.index(lang)
            entry = to_tensor(sample, lang_idx, eos_idx, shifted)
            return entry

        first  = get(self.first, idy, shifted=False)
        second = get(self.second, idy, shifted=True)
        return (first, second)

    @property
    def lengths(self):
        return self.first.metadata.lens

