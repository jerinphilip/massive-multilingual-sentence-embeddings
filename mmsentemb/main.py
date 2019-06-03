from argparse import ArgumentParser
import os
import sys
from fairseq.data.dictionary import Dictionary
import ilmulti as ilm
from .models import Encoder, Decoder, EmbeddingModel
from .data import ParallelDataset
from torch.utils.data import DataLoader

def add_args(parser):
    parser.add_argument('--dict_path', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--source_lang', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--target_lang', type=str, required=True)

if __name__ == '__main__':
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    dictionary = Dictionary.load(args.dict_path)
    tokenizer = ilm.sentencepiece.SentencePieceTokenizer()
    dataset = ParallelDataset(
            (args.source, args.source_lang),
            (args.target, args.target_lang),
            tokenizer,
            dictionary
    )

    loader = DataLoader(dataset, collate_fn=dataset.collate(), batch_size=4)
    for sample in loader:
        for key in sample:
            print(key, sample[key].size())
        break


