import os
from .data import ParallelDataset, collate, ShardedBatchIterator
from .trainer import Trainer
from fairseq.data.dictionary import Dictionary
import ilmulti as ilm
from torch import optim
from itertools import permutations
from collections import namedtuple, OrderedDict

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
        # def _get_individual(source, target):
        #     return ParallelDataset(source, target, 
        #             self.tokenizer, self.dictionary)
        # pairs = [
        #     Corpus(args.source, args.source_lang), 
        #     Corpus(args.target, args.target_lang)
        # ]

        def _get(args, split):
            split_prefix = os.path.join(args.prefix, split)
            source = '{}.{}'.format(split_prefix, args.source_lang)
            target = '{}.{}'.format(split_prefix, args.target_lang)
            source_corpus = Corpus(source, args.source_lang)
            target_corpus = Corpus(target, args.target_lang)
            return ParallelDataset(source_corpus, target_corpus, 
                    self.tokenizer, self.dictionary)

        self.dataset = OrderedDict()
        splits = ["train", "dev", "test"]
        for split in splits[:2]:
            self.dataset[split] = _get(args, split)

        return self.dataset

    def get_loader(self):
        args = self.args
        loader = OrderedDict()
        for split in self.dataset:
            itr = ShardedBatchIterator(
                self.dataset[split], 
                collate_fn=collate(self.dictionary), 
                max_tokens=args.max_tokens,
                shard_idx=args.distributed_rank,
                num_shards=args.distributed_world_size,
                shuffle=True
            )
            loader[split] = itr
        return loader
