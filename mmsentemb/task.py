from .data import ParallelDataset, collate, EpochBatchIterator
from .trainer import Trainer
from fairseq.data.dictionary import Dictionary
from .models import EmbeddingModel
import ilmulti as ilm
from torch import optim

class JointSpaceLearningTask:
    def __init__(self, args):
        self.args = args
        self.dictionary = Dictionary.load(args.dict_path)
        self.tokenizer = ilm.sentencepiece.SentencePieceTokenizer()


    def load_dataset(self):
        args = self.args
        self.datasets = [
            ParallelDataset(
                (args.source, args.source_lang),
                (args.target, args.target_lang),
                self.tokenizer,
                self.dictionary
            ),
            ParallelDataset(
                (args.target, args.target_lang),
                (args.source, args.source_lang),
                self.tokenizer,
                self.dictionary
            ),
        ]

    def get_loader(self):
        loaders = [
                EpochBatchIterator(
                    dataset, 
                    collate_fn=collate(self.dictionary), 
                    max_tokens=self.args.max_tokens,
                    shuffle=False
                )
                for dataset in self.datasets
        ]
        return loaders

    def build_trainer(self):
        self.model = EmbeddingModel.build(self.args, self.dictionary)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.logger = None
        self.trainer = Trainer(self.args, self.model, self.optimizer, self.logger)
