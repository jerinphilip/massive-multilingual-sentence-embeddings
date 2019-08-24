import os
import yaml
from .data import ParallelDataset, collate, ShardedBatchIterator
from .data import MultiwayDataset
from .trainer import Trainer
from fairseq.data.dictionary import Dictionary
import ilmulti as ilm
from torch import optim
from itertools import permutations
from collections import namedtuple, OrderedDict
from mmse.data.config.utils import pairs_select

class JointSpaceLearningTask:
    def __init__(self, args):
        self.args = args
        self.data = self.load_dataconfig(self.args.config_file)
        self.dictionary = Dictionary.load(self.data['dictionary']['src'])
        self.tokenizer = ilm.sentencepiece.SentencePieceTokenizer()

    def setup_task(self):
        self.load_dataset()

    def load_dataconfig(self, config_file):
        with open(config_file) as fp:
            content = fp.read()
            data = yaml.load(content)
            return data

    def load_dataset(self):
        def _get(split):
            pairs = pairs_select(self.data['corpora'], split)
            dataset = MultiwayDataset(
                pairs, self.tokenizer, 
                self.dictionary
            )
            return dataset

        self.dataset = OrderedDict()
        splits = ["train", "dev", "test"]
        for split in splits[:2]:
            self.dataset[split] = _get(split)

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
                shuffle=(True if split is 'train' else False)
            )
            loader[split] = itr
        return loader
