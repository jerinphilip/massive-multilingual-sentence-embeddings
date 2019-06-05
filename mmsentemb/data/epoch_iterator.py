import os
from torch.utils.data import Dataset
import torch
import ilmulti as ilm
from collections import namedtuple
import random
from copy import deepcopy


class EpochBatchIterator:
    def __init__(self, dataset, collate_fn, max_tokens, shuffle=False):
        self.dataset = dataset
        self.collate = collate_fn
        self.max_tokens = max_tokens
        self._preconditions()
        self.shuffle = shuffle

    def _preconditions(self):
        def chunks(lengths):
            idx  = -1
            N = len(lengths)
            while idx < N - 1:
                total = 0
                start = idx + 1
                while total < self.max_tokens and idx < N-1:
                    idx += 1
                    total += lengths[idx]

                if total > self.max_tokens:
                    total -= lengths[idx]
                    idx -= 1

                yield (start, idx)

        lengths = self.dataset.export["lens"] 
        print("Obtaining batches")
        self.batches = list(chunks(lengths))
        self.toks = [sum(lengths[s:v]) for s, v in self.batches]
        print("Obtained {} batches".format(len(self.batches)))

    def __iter__(self):
        self.idx = -1
        if self.shuffle:
            random.shuffle(self.batches)
        return self

    def __next__(self):
        self.idx += 1
        if self.idx < len(self.batches):
            s, v = self.batches[self.idx]
            idxs = list(range(s, v))
            samples = [self.dataset[idx] for idx in idxs]
            return self.collate(samples)
        else:
            raise StopIteration

    def __len__(self):
        return len(self.batches)

