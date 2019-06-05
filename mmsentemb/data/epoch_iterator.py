import os
from torch.utils.data import Dataset
import torch
import ilmulti as ilm
from collections import namedtuple
import random
from copy import deepcopy


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
    def __init__(self, dataset, collate_fn, max_tokens, shard_idx, num_shards, shuffle=False):
        random.seed(num_shards)
        lengths = dataset.export["lens"]
        _batches = batches(lengths, max_tokens)
        if shuffle:
            random.shuffle(_batches)

        while len(_batches) % num_shards != 0:
            random_batch = random.choice(batches)
            _batches.append(random_batch)

        self._num_batches  = len(_batches) // num_shards

        start = shard_idx*self._num_batches
        end = (shard_idx+1)*self._num_batches
        self._batches = _batches[start:end]
        self._prefetch_batches(dataset, collate_fn)

    def _prefetch_batches(self, dataset, collate_fn):
        self.batches = []
        for s, v in self._batches:
            idxs = list(range(s, v))
            samples = [dataset[idx] for idx in idxs]
            batch = collate_fn(samples)
            self.batches.append(batch)

    def __iter__(self):
        return self.batches.__iter__()
        

    def __len__(self):
        return len(self.batches)


