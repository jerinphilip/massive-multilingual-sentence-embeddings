import os
import sys
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch import optim
from fairseq.data.dictionary import Dictionary

import ilmulti as ilm
from .models import EmbeddingModel
from .data import ParallelDataset
from .trainer import Trainer

def add_args(parser):
    parser.add_argument('--dict_path', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--source_lang', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--target_lang', type=str, required=True)
    parser.add_argument('--encoder_embedding_dim', type=int, default=512)
    parser.add_argument('--encoder_hidden_size', type=int, default=512)
    parser.add_argument('--encoder_num_layers', type=int, default=3)
    parser.add_argument('--encoder_bidirectional', type=bool, default=True)
    parser.add_argument('--decoder_embedding_dim', type=int, default=512)

    # Set this to 2 * encoder
    parser.add_argument('--decoder_hidden_size', type=int, default=1024)
    parser.add_argument('--decoder_num_layers', type=int, default=3)
    parser.add_argument('--decoder_bidirectional', type=bool, default=False)
    parser.add_argument('--decoder_output_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=64)

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

    loader = DataLoader(dataset, collate_fn=dataset.collate(), batch_size=args.batch_size)
    model = EmbeddingModel.build(args, dictionary)
    optimizer = optim.Adam(model.parameters())
    trainer = Trainer(model, optimizer)
    for batch_idx, batch in enumerate(loader):
        loss = trainer.run_update(batch)
        if batch_idx % 10 == 0:
            print(loss)


