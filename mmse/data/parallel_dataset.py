import os
import lmdb
import torch
import pickle
import random
import ilmulti as ilm
from torch.utils.data import Dataset
from collections import namedtuple
from copy import deepcopy
from .lmdb import LMDBCorpus

def to_tensor(sample, lang_idx, eos_idx, shifted=False):
    idxs = deepcopy(sample)
    idxs.append(eos_idx)
    if shifted:
        idxs.insert(0, eos_idx)
    return torch.LongTensor(idxs), torch.LongTensor([lang_idx])

class ParallelDataset(Dataset):
    def __init__(self, first, second, tokenizer, dictionary):
        self.tokenizer = tokenizer
        self.dictionary = dictionary
        self.first = LMDBCorpus(first)
        self.second = LMDBCorpus(second)

    def __len__(self):
        return self.first.num_samples

    def __getitem__(self, idx):
        idy = self.first.idxs[idx]
        eos_idx = self.dictionary.eos()

        def get(holding, idy, shifted):
            sample = holding[idy]
            lang = ilm.utils.language_token(holding.corpus.lang)
            lang_idx = self.dictionary.index(lang)
            entry = to_tensor(sample, lang_idx, eos_idx, shifted)
            return entry

        first  = get(self.first, idy, shifted=False)
        second = get(self.second, idy, shifted=True)
        return (first, second)

    @property
    def lengths(self):
        return self.first.lengths
