import lmdb
import numpy as np
import pickle
from collections import namedtuple
from fairseq.data.dictionary import Dictionary
import ilmulti as ilm
from tqdm import tqdm

Corpus = namedtuple('Corpus', 'path lang')
Meta = namedtuple('Meta', 'idxs lengths')


class LMDBCorpus:
    def __init__(self, corpus):
        map_size = 1 << 40
        self.corpus = corpus
        path = '{}.processed.lmdb'.format(corpus.path)
        self.env = lmdb.open(path, map_size=map_size)

    def __getitem__(self, key):
        assert(isinstance(key, int))
        key = '{}'.format(key).encode("ascii")
        with self.env.begin() as txn:
            record = txn.get(key)
            data = pickle.load(record)
            return data



class LMDBCorpusReader:
    def __init__(self, corpus):
        map_size = 1 << 40
        self.corpus = corpus
        path = '{}.processed.lmdb'.format(corpus.path)
        self.env = lmdb.open(path, map_size=map_size)

    def debug(self):
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                print(key)


class LMDBCorpusWriter:
    def __init__(self, corpus):
        map_size = 1 << 40
        self.corpus = corpus
        path = '{}.processed.lmdb'.format(corpus.path)
        self.env = lmdb.open(path, map_size=map_size)

        # Write corpus
        corpus = pickle.dumps(self.corpus._asdict)
        self._set("corpus", corpus)

    def write_metadata(self, metadata):
        idxs = pickle.dumps(np.array(metadata.idxs), protocol=0)
        lengths = pickle.dumps(np.array(metadata.lengths), protocol=0)
        self._set("idxs", idxs)
        self._set("lengths", lengths)
        # Total samples
        num_samples = '{}'.format(len(metadata.lengths)).encode("ascii")
        self._set("num_samples", num_samples)

    def _set(self, key, val):
        with self.env.begin(write=True) as txn:
            key = key.encode("ascii")
            txn.put(key, val)

    def write(self, idx, sample):
        sample = pickle.dumps(np.array(sample), protocol=0)
        self._set(idx, sample)
        
def idxs_lengths(tokenizer, _file):
    lines = open(_file).read().splitlines()
    lengths = []
    for line in tqdm(lines, desc='idxs-lens', leave=True):
        lang, tokens = tokenizer(line)
        _len = len(tokens)
        lengths.append(_len)

    N = len(lines)
    lengths = list(zip(range(N), lengths))
    lengths = sorted(lengths, key = lambda x: x[1])
    idxs, lengths = list(zip(*lengths))
    return idxs, lengths


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--lang', type=str, required=True)
    parser.add_argument('--dict', type=str, required=True)
    args = parser.parse_args()
    tokenizer = ilm.sentencepiece.SentencePieceTokenizer()
    dictionary = Dictionary.load(args.dict)

    corpus = Corpus(args.path, args.lang)
    # reader = LMDBCorpusReader(corpus)
    # reader.debug()
    writer = LMDBCorpusWriter(corpus)
    # idxs, lengths = idxs_lengths(tokenizer, corpus.path)
    # metadata = Meta(idxs, lengths)
    # writer.write_metadata(metadata)
    def _get(line):
        _lang, tokens = tokenizer(line)
        idxs = [dictionary.index(token) for token in tokens]
        return idxs

    content = open(corpus.path).read().splitlines()
    for idx, line in tqdm(enumerate(content), total=len(content)):
        key = '{}'.format(idx)
        tensor = _get(line)
        val = tensor
        # val = pickle.dumps(tensor, protocol=0)
        writer.write(key, val)
        # print(line)

