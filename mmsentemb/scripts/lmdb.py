import lmdb
import numpy as np
import pickle
from collections import namedtuple
from fairseq.data.dictionary import Dictionary
import ilmulti as ilm
from tqdm import tqdm
import os
from ..data.lmdb import LMDBCorpus, LMDBCorpusWriter

Corpus = namedtuple('Corpus', 'path lang')
Meta = namedtuple('Meta', 'idxs lengths')
        
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
    idxs, lengths = idxs_lengths(tokenizer, corpus.path)
    metadata = Meta(idxs, lengths)
    writer.write_metadata(metadata)
    # def _get(line):
    #     _lang, tokens = tokenizer(line)
    #     idxs = [dictionary.index(token) for token in tokens]
    #     return idxs

    # content = open(corpus.path).read().splitlines()
    # for idx, line in tqdm(enumerate(content), total=len(content)):
    #     key = '{}'.format(idx)
    #     tensor = _get(line)
    #     val = tensor
    #     # val = pickle.dumps(tensor, protocol=0)
    #     writer.write(key, val)
    #     # print(line)

