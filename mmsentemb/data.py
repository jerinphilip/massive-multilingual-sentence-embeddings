from torch.utils.data import Dataset
import torch
import ilmulti as ilm
from collections import namedtuple

class ParallelDataset:
    def __init__(self, first, second, tokenizer, dictionary):
        self.tokenizer = tokenizer
        self.dictionary = dictionary
        self.first = self._load(first)
        self.second = self._load(second)
        assert(len(self.first.lines) == len(self.second.lines))

    def _load(self, one):
        One = namedtuple('One', 'lines lang')
        path, lang = one
        content = open(path).read().splitlines()
        return One(content, lang)

    def __len__(self):
        return len(self.first.lines)

    def __getitem__(self, idx):
        def _get(one, idx):
            line = one.lines[idx]
            lang, tokens = self.tokenizer(line)
            idxs = [self.dictionary.index(token) for token in tokens]
            lang_token = ilm.utils.language_token(one.lang)
            lang_idx = self.dictionary.index(lang_token)
            return (torch.LongTensor(idxs), torch.LongTensor(lang_idx))

        first = _get(self.first, idx)
        second = _get(self.second, idx)

        return (first, second)

    def collate(self):
        def _collate(samples):
            def fseq_length(sample):
                first, second = sample
                idxs, lang_idxs = first
                return idxs.size(0)

            samples = sorted(samples, key=fseq_length)
            first, second = list(zip(*samples))
            fidxs, flang_idxs = list(zip(*first))
            sidxs, slang_idxs = list(zip(*second))
            fseq_lengths = [f.size(0) for f in fidxs]

            fidxs = torch.nn.utils.rnn.pad_sequence(fidxs, 
                    padding_value=self.dictionary.pad())
            sidxs = torch.nn.utils.rnn.pad_sequence(fidxs, 
                    padding_value=self.dictionary.pad())

            flang_idxs = torch.stack(flang_idxs)
            slang_idxs = torch.stack(slang_idxs)

            return (fidxs, fseq_lengths, flang_idxs, sidxs, slang_idxs)
        return _collate
