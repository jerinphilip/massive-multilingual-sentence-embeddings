from .data import ParallelDataset, collate, ShardedBatchIterator
from .trainer import Trainer
from fairseq.data.dictionary import Dictionary
import ilmulti as ilm
from torch import optim
from itertools import permutations
from collections import namedtuple

class JointSpaceLearningTask:
    def __init__(self, args):
        self.args = args
        self.dictionary = Dictionary.load(args.dict_path)
        self.tokenizer = ilm.sentencepiece.SentencePieceTokenizer()

    def setup_task(self):
        self.load_dataset()

    def load_dataset(self):
        args = self.args
        Corpus = namedtuple('Corpus', 'path lang')
        def _get_individual(source, target):
            return ParallelDataset(source, target, 
                    self.tokenizer, self.dictionary)
        pairs = [
            Corpus(args.source, args.source_lang), 
            Corpus(args.target, args.target_lang)
        ]

        self.datasets = [ 
            _get_individual(p1, p2) 
            for p1, p2 in permutations(pairs) 
        ]

    def get_loader(self):
        args = self.args
        loaders = [
                ShardedBatchIterator(
                    dataset, 
                    collate_fn=collate(self.dictionary), 
                    max_tokens=self.args.max_tokens,
                    shard_idx=args.distributed_rank,
                    num_shards=args.distributed_world_size,
                    shuffle=True
                )
                for dataset in self.datasets
        ]
        return loaders
