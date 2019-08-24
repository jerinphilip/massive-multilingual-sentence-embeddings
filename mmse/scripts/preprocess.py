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
        
def main(args):
    tokenizer = ilm.sentencepiece.SentencePieceTokenizer()
    dictionary = Dictionary.load(args.dict)
    corpus = Corpus(args.path, args.lang)
    corpus_writer = LMDBCorpusWriter(corpus, dictionary, tokenizer)
    corpus_writer.build_corpus(corpus, dictionary, tokenizer)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--lang', type=str, required=True)
    parser.add_argument('--dict', type=str, required=True)
    args = parser.parse_args()
    main(args)
