import os
import lmdb
import torch
import pickle
import random
import itertools
import ilmulti as ilm
from torch.utils.data import Dataset, ConcatDataset
from collections import namedtuple
from copy import deepcopy
from .lmdb import LMDBCorpus
import numpy as np

_flyweight = {}

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
        self.first = self._maybe_load(first, dictionary, tokenizer)
        self.second = self._maybe_load(second, dictionary, tokenizer)
        self.lengths = np.maximum(self.first.lengths, self.second.lengths)

    def _maybe_load(self, corpus, dictionary, tokenizer):
        if corpus.path not in _flyweight:
            _flyweight[corpus.path] = LMDBCorpus.build(corpus, dictionary, tokenizer)
        return _flyweight[corpus.path]

    def __len__(self):
        return self.first.num_samples

    def __getitem__(self, idx):
        eos_idx = self.dictionary.eos()

        def get(holding, idy, shifted):
            tokens = holding[idy]
            sample = [self.dictionary.index(token) for token in tokens]
            lang = ilm.utils.language_token(holding.corpus.lang)
            lang_idx = self.dictionary.index(lang)
            entry = to_tensor(sample, lang_idx, eos_idx, shifted)
            return entry

        first  = get(self.first, idx, shifted=False)
        second = get(self.second, idx, shifted=True)
        return (first, second)


class MultiwayDataset:
    def __init__(self, pairs, tokenizer, dictionary):
        self.datasets = [
            ParallelDataset(fst, snd, tokenizer, dictionary)
            for fst, snd in pairs
        ]
        self.concatenated = ConcatDataset(self.datasets)
        self.lengths = list(itertools.chain(
            *[dataset.lengths for dataset in self.datasets]
        ))

        self.indices = np.argsort(self.lengths)

    def __getitem__(self, idx):
        # idy = self.indices[idx]
        # print(idy, idx, self.lengths[idx])
        return self.concatenated.__getitem__(idx)

    def __len__(self):
        return self.concatenated.__len__()
