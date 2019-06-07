import os
import lmdb
import pickle
import numpy as np

class LMDBCorpusWriter:
    def __init__(self, corpus):
        map_size = LMDBCorpusWriter.corpus_map_size(corpus)
        self.corpus = corpus
        path = '{}.processed.lmdb'.format(corpus.path)
        self.env = lmdb.open(path, map_size=map_size)

        # Write corpus
        corpus = pickle.dumps(self.corpus._asdict)
        self._set("corpus", corpus)

    @staticmethod
    def corpus_map_size(corpus):
        scale = 10
        size = int(scale*os.path.getsize(corpus.path))
        return size

    def serialize(self, obj):
        BEST_COMPRESS = 4
        return pickle.dumps(obj, protocol=BEST_COMPRESS)

    def write_metadata(self, metadata):
        idxs = self.serialize(np.array(metadata.idxs, dtype=np.uint32))
        lengths = self.serialize(np.array(metadata.lengths, dtype=np.uint32))
        self._set("idxs", idxs)
        self._set("lengths", lengths)
        num_samples = '{}'.format(len(metadata.lengths)).encode("ascii")
        self._set("num_samples", num_samples)

    def _set(self, key, val):
        with self.env.begin(write=True) as txn:
            key = key.encode("ascii")
            txn.put(key, val)

    def write(self, idx, sample):
        sample = self.serialize(np.array(sample, dtype=np.uint32))
        self._set(idx, sample)

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
        _key = '{}'.format(key)
        record = self._get_value(_key)
        unpickled = pickle.loads(record)
        return unpickled.tolist()
