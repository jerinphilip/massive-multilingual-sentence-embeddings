import os
import torch
import random
import ilmulti as ilm
from copy import deepcopy
from collections import namedtuple
from torch.utils.data import Dataset

def batches(lengths, max_tokens):
    def chunks(lengths):
        idx  = -1
        N = len(lengths)
        while idx < N - 1:
            total = 0
            start = idx + 1
            while total < max_tokens and idx < N-1:
                idx += 1
                total += lengths[idx]

            if total > max_tokens:
                total -= lengths[idx]
                idx -= 1

            yield (start, idx)
    return list(chunks(lengths))


class ShardedBatchIterator:
    def __init__(self, dataset, collate_fn, max_tokens, 
                        shard_idx, num_shards, shuffle=False):
        random.seed(num_shards)
        lengths = dataset.lengths
        _batches = batches(lengths, max_tokens)
        if shuffle:
            random.shuffle(_batches)

        while len(_batches) % num_shards != 0:
            random_batch = random.choice(_batches)
            _batches.append(random_batch)

        self._num_batches  = len(_batches) // num_shards

        start = shard_idx*self._num_batches
        end = (shard_idx+1)*self._num_batches
        self.dataset = dataset
        self.collate = collate_fn
        self._batches = _batches[start:end]
        self._cached_batches = []

    def _prefetch_batches(self, dataset, collate_fn):
        self.batches = []
        for s, v in self._batches:
            idxs = list(range(s, v+1))
            samples = [dataset[idx] for idx in idxs]
            batch = collate_fn(samples)
            self.batches.append(batch)


    def __iter__(self):
        if self._cached_batches:
            return self._cached_batches.__iter__()
        self.batch_idx = 0
        return self

        
    def __len__(self):
        return len(self._batches)

    def __next__(self):
        if self.batch_idx >= len(self._batches):
            raise StopIteration
        
        s, v = self._batches[self.batch_idx] 
        idxs = list(range(s, v+1))
        samples = [self.dataset[idx] for idx in idxs]
        batch = self.collate(samples)
        self.batch_idx = self.batch_idx + 1
        self._cached_batches.append(batch)
        return batch


