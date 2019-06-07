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
        
def main(args):
    tokenizer = ilm.sentencepiece.SentencePieceTokenizer()
    dictionary = Dictionary.load(args.dict)

    corpus = Corpus(args.path, args.lang)
    writer = LMDBCorpusWriter(corpus)
    def _get(line):
        _lang, tokens = tokenizer(line)
        idxs = [dictionary.index(token) for token in tokens]
        length = len(tokens)
        return idxs, length

    content = open(corpus.path).read().splitlines()
    lengths = []
    pbar = tqdm(
        enumerate(content), 
        total=len(content),
        ascii="#",
        ncols=200
    )

    for idx, line in pbar:
        key = '{}'.format(idx)
        token_idxs, length = _get(line)
        writer.write(key, token_idxs)
        lengths.append(length)

    sample_idxs = range(len(lengths))
    zipped = list(zip(sample_idxs, lengths))
    pairs_sorted = sorted(zipped, key=lambda x: x[1])
    sample_idxs, lengths = list(zip(*pairs_sorted))
    metadata = Meta(sample_idxs, lengths)
    writer.write_metadata(metadata)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--lang', type=str, required=True)
    parser.add_argument('--dict', type=str, required=True)
    args = parser.parse_args()
    main(args)
