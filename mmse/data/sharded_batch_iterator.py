import os
import torch
import random
import ilmulti as ilm
from copy import deepcopy
from collections import namedtuple
from torch.utils.data import Dataset

class ShardedBatchIterator:
    def __init__(self, dataset, collate_fn, max_tokens, 
                        shard_idx, num_shards, shuffle=False):

        self.PRNG = random.Random(num_shards)
        self.shuffle = shuffle
        self.shard_idx = shard_idx
        self.num_shards = num_shards
        self.dataset = dataset
        self.collate = collate_fn

        # Batch properties
        _batches = self.precompute_batch_idxs(dataset.lengths, max_tokens)
        while len(_batches) % num_shards != 0:
            random_batch = self.PRNG.choice(_batches)
            _batches.append(random_batch)

        self.batches = _batches
        self._num_batches  = len(_batches) // num_shards

    def precompute_batch_idxs(self, lengths, max_tokens):
        N = len(lengths)
        idxs = list(range(N))

        # Sort lengths
        paired = zip(idxs, lengths)
        paired = sorted(paired, key=lambda x: x[1])

        batches = []

        current_length = 0
        batch = []
        for idx, length in paired:
            if current_length + length <= max_tokens:
                batch.append(idx)
                current_length += length
            else:
                batches.append(batch)
                batch = []
                current_length = 0

        if batch:
            batches.append(batch)
        return batches

    def _reshuffle(self):
        if self.shuffle:
            self.PRNG.shuffle(self.batches)

        start = self.shard_idx*self._num_batches
        end = (self.shard_idx+1)*self._num_batches
        self._batches = self.batches[start:end]

        # Debug
        # self.debug()

    def debug(self):
        with open("/scratch/batches.{}".format(self.shard_idx), 'w+') as fp:
            for batch in self.batches:
                s = '|'.join(map(str, batch))
                fp.write(s + '\n')
        with open("/scratch/lengths.{}".format(self.shard_idx), 'w+') as fp:
            for length in self.dataset.lengths:
                fp.write(str(length) + '\n')

        exit()


    def __iter__(self):
        self.batch_idx = 0
        self._reshuffle()
        return self

        
    def __len__(self):
        return self._num_batches

    def __next__(self):
        if self.batch_idx >= self._num_batches:
            raise StopIteration
        
        # TODO(jerin): Fix this.
        try:
            batch = self._get_batch(self.batch_idx)
        except:
            # Debug
            batch_idx = (self.batch_idx + 1)%self._num_batches
            batch = self._get_batch(batch_idx)

        self.batch_idx = self.batch_idx + 1
        return batch

    def _get_batch(self, batch_idx):
        idxs  = self._batches[batch_idx] 
        samples = []
        for idx in idxs:
            sample = self.dataset[idx]
            samples.append(sample)

        batch = self.collate(samples)
        return batch


