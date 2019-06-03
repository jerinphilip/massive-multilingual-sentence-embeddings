from argparse import ArgumentParser
import os
import sys
from fairseq.data.dictionary import Dictionary
import ilmulti as ilm
from .models import Encoder, Decoder, EmbeddingModel

def add_args(parser):
    parser.add_argument('--dict_path', type=str, required=True)

if __name__ == '__main__':
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    dictionary = Dictionary.load(args.dict_path)
