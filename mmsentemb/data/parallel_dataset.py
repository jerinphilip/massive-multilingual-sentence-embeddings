import os
from torch.utils.data import Dataset
import torch
import ilmulti as ilm
from collections import namedtuple
import random
from copy import deepcopy
from .corpus_flyweight import CORPUS_REGISTRY
import lmdb
import pickle

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
        # self.first = CORPUS_REGISTRY.get(first, tokenizer, dictionary)
        # self.second = CORPUS_REGISTRY.get(first, tokenizer, dictionary)
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

class LMDBCorpus:
    def __init__(self, corpus):
        map_size = 1 << 40
        self.corpus = corpus
        path = '{}.processed.lmdb'.format(corpus.path)
        self.env = lmdb.open(path, map_size=map_size)
        self._init_metadata()

    def _init_metadata(self):
        idxs = self._get_value("idxs")
        lengths = self._get_value("lengths")
        num_samples = self._get_value("num_samples")

        self.idxs = pickle.loads(idxs).tolist()
        self.lengths = pickle.loads(lengths).tolist()
        self.num_samples = int(num_samples.decode("ascii"))
        

    def _get_value(self, key):
        key = key.encode("ascii")
        with self.env.begin() as txn:
            record = txn.get(key)
        return record

    def __getitem__(self, key):
        # assert(isinstance(key, int))
        _key = '{}'.format(key)
        record = self._get_value(_key)
        unpickled = pickle.loads(record)
        return unpickled.tolist()
