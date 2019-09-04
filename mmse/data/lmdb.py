import os
import lmdb
# import pickle
import numpy as np
from tqdm import tqdm
from copy import deepcopy

class LMDBCorpusWriter:
    def __init__(self, corpus):
        map_size = LMDBCorpusWriter.corpus_map_size(corpus)
        self.corpus = corpus
        path = LMDBCorpusWriter.path(corpus)
        self.env = lmdb.open(path, map_size=map_size)

    @staticmethod
    def path(corpus):
        _path = '{}.mmse.lmdb'.format(corpus.path)
        return _path

    @staticmethod
    def corpus_map_size(corpus):
        scale = 20
        size = int(scale*os.path.getsize(corpus.path))
        return size
    
    def write_metadata(self, lengths):
        # Set __len__
        num_samples = len(lengths)
        num_samples = '{}'.format(num_samples).encode("ascii")
        self._set("num_samples", num_samples)

        # # Set Idxs
        # idxs = np.array(idxs, dtype=np.int32).tobytes()
        # self._set("idxs", idxs)

        # Set lengths
        lengths = np.array(lengths, dtype=np.int32).tobytes()
        self._set("lengths", lengths)


    def _set(self, key, val):
        with self.env.begin(write=True) as txn:
            key = key.encode("ascii")
            txn.put(key, val)

    def write_sample(self, idx, sample):
        self._set(idx, sample.encode())

    def build_corpus(self, corpus, tokenizer):
        writer = LMDBCorpusWriter(corpus)
        def _get(line):
            _lang, tokens = tokenizer(line, corpus.lang)
            length = len(tokens)
            tokenized = ' '.join(tokens)
            return tokenized, length

        content = open(corpus.path).read().splitlines()
        lengths = []
        pbar = tqdm(
            enumerate(content), total=len(content),
            ascii="#", ncols=200, miniters=1000, dynamic_ncols=True,
            desc=corpus.path
        )

        # pbar = enumerate(content)

        for idx, line in pbar:
            key = '{}'.format(idx)
            tokenized, length = _get(line)
            writer.write_sample(key, tokenized)
            lengths.append(length)

        # sample_idxs = range(len(lengths))
        # zipped = list(zip(sample_idxs, lengths))
        # pairs_sorted = sorted(zipped, key=lambda x: x[1])
        # idxs, lengths = list(zip(*pairs_sorted))
        writer.write_metadata(lengths)

    def close(self):
        self._set("completed", "true")
        self.env.close()

class LMDBCorpus:
    def __init__(self, corpus):
        self.corpus = corpus
        map_size = LMDBCorpusWriter.corpus_map_size(corpus)
        path = LMDBCorpusWriter.path(corpus)
        # path = '{}.processed.lmdb'.format(corpus.path)
        self.env = lmdb.open(path, map_size=map_size, readonly=True)
        self._init_metadata()

    def _init_metadata(self):
        # idxs = self._get_value("idxs")
        lengths = self._get_value("lengths")
        num_samples = self._get_value("num_samples")

        # self.idxs = np.frombuffer(idxs, dtype=np.int32)
        self.lengths = np.frombuffer(lengths, dtype=np.int32)
        self.num_samples = int(num_samples.decode("ascii"))
        

    def _get_value(self, key):
        key = key.encode("ascii")
        with self.env.begin() as txn:
            record = txn.get(key)
        return record

    def __getitem__(self, key):
        _key = '{}'.format(key)
        record = self._get_value(_key)
        tokenized = record.decode('utf-8')
        tokens = tokenized.split()
        return tokens

    @classmethod
    def build(cls, corpus, tokenizer):
        cache_path = LMDBCorpusWriter.path(corpus)
        try:
            return cls(corpus)
        except:
            writer = LMDBCorpusWriter(corpus) 
            writer.build_corpus(corpus, tokenizer)
            writer.close()
            return cls(corpus)

