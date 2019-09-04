import os
import lmdb
import numpy as np
import pickle
from collections import namedtuple
from fairseq.data.dictionary import Dictionary
import ilmulti as ilm
from tqdm import tqdm
from ..data.lmdb import LMDBCorpus, LMDBCorpusWriter
from multiprocessing import Pool
from mmse.data.config.utils import pairs_select
import yaml


Corpus = namedtuple('Corpus', 'path lang')
        
def build_corpus(corpus):
    tokenizer = ilm.sentencepiece.SentencePieceTokenizer()
    corpus_writer = LMDBCorpusWriter(corpus)
    corpus_writer.build_corpus(corpus, tokenizer)

def unique_corpora(config_file):
    def load_dataconfig(config_file):
        with open(config_file) as fp:
            content = fp.read()
            data = yaml.load(content)
            return data

    data = load_dataconfig(config_file)
    
    corpora = []
    for split in ['train', 'dev', 'test']:
        pairs = pairs_select(data['corpora'], split)
        srcs, tgts = list(zip(*pairs))
        corpora.extend(srcs)
        corpora.extend(tgts)

    corpora = list(set(corpora))
    corpora = sorted(corpora, key=lambda x: x.path)
    return corpora

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    args = parser.parse_args()
    corpora = unique_corpora(args.config_file)
    cores = os.cpu_count()
    print(
        "Building {count} corpus over {cores} cores"
            .format(cores=cores, count=len(corpora))
    )
    pool = Pool(processes=cores)
    pool.map(build_corpus, corpora)
    # pool.map_async(build_corpus, corpora)
    # pool.close()
    # pool.join()

