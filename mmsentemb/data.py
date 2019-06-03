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
        def _get(one, idx, eos_end=True):
            line = one.lines[idx]
            lang, tokens = self.tokenizer(line)
            idxs = [self.dictionary.index(token) for token in tokens]
            if eos_end:
                idxs.append(self.dictionary.eos())
            else:
                idxs.insert(0, self.dictionary.eos())
            lang_token = ilm.utils.language_token(one.lang)
            lang_idx = self.dictionary.index(lang_token)
            return (torch.LongTensor(idxs), torch.LongTensor([lang_idx]))

        first = _get(self.first, idx)
        second = _get(self.second, idx)

        return (first, second)

    def collate(self):
        def _collate(samples):
            def fseq_length(sample):
                first, second = sample
                idxs, lang_idxs = first
                return idxs.size(0)

            samples = sorted(samples, key=fseq_length, reverse=True)
            first, second = list(zip(*samples))

            def _extract(one):
                idxs, lang_idxs = list(zip(*one))
                seq_lengths = [sample.size(0) for sample in idxs]
                seq_lengths = torch.LongTensor(seq_lengths)
                idxs = torch.nn.utils.rnn.pad_sequence(idxs, 
                        padding_value=self.dictionary.pad())

                idxs = idxs.transpose(0, 1).contiguous()
                lang_idxs = torch.stack(lang_idxs, dim=0)
                return idxs, lang_idxs, seq_lengths

            fidxs, flang_idxs, fseq_lengths = _extract(first)
            sidxs, slang_idxs, sseq_lengths = _extract(second)

            export = {
                "srcs": fidxs,
                "tgts": sidxs,
                "src_lens": fseq_lengths,
                "src_langs": flang_idxs,
                "tgt_langs": slang_idxs
            }
            return export
        return _collate
