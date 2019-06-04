import os
from torch.utils.data import Dataset
import torch
import ilmulti as ilm
from collections import namedtuple
import random
from copy import deepcopy

class ParallelDataset:
    def __init__(self, first, second, tokenizer, dictionary):
        self.tokenizer = tokenizer
        self.dictionary = dictionary
        _first_file, _ = first
        export = compute_tokenized_lengths(_first_file, tokenizer)
        self.export = export
        idxs = export["idxs"]
        self.first = self._load(first, idxs=idxs)
        self.second = self._load(second, idxs=idxs)
        assert(self.first.length == self.second.length)


    def _load(self, one, idxs=None):
        One = namedtuple('One', 'tensors length')
        raw_tensors = self._preload(one)
        if idxs is not None:
            tensors = [raw_tensors[i] for i in idxs]
        else:
            tensors = raw_tensors
        return One(tensors, len(tensors))

    def _preload(self, one):
        path, lang = one
        save_path = '{}.tensors'.format(path)
        if os.path.exists(save_path):
            print("{} exists. loading".format(save_path))
            return torch.load(save_path)

        else:
            print("{} does not exist. computing and saving.".format(save_path))
            def _get(line):
                # line = content[idx]
                _lang, tokens = self.tokenizer(line)
                idxs = [self.dictionary.index(token) for token in tokens]
                lang_token = ilm.utils.language_token(lang)
                lang_idx = self.dictionary.index(lang_token)
                return (idxs, lang_idx)

            content = open(path).read().splitlines()
            tensors = [_get(line) for line in content]
            torch.save(tensors, save_path)
            return tensors

    def __len__(self):
        return self.first.length

    def __getitem__(self, idx):
        def to_tensor(sample, eos_end=True):
            idxs, lang_idx = deepcopy(sample)
            if eos_end:
                idxs.append(self.dictionary.eos())
            else:
                idxs.insert(0, self.dictionary.eos())
                idxs.append(self.dictionary.eos())
            return torch.LongTensor(idxs), torch.LongTensor([lang_idx])


        first = to_tensor(self.first.tensors[idx], eos_end=True)
        second = to_tensor(self.second.tensors[idx], eos_end=False)
        return (first, second)

def collate(dictionary):
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
                    padding_value=dictionary.pad())

            idxs = idxs.transpose(0, 1).contiguous()
            lang_idxs = torch.cat(lang_idxs, dim=0)
            return idxs, lang_idxs, seq_lengths

        fidxs, flang_idxs, fseq_lengths = _extract(first)
        sidxs, slang_idxs, sseq_lengths = _extract(second)

        batch_size = fidxs.size(0)
        export = {
            "srcs": fidxs,
            "tgts": sidxs,
            "src_lens": fseq_lengths,
            "src_langs": flang_idxs,
            "tgt_langs": slang_idxs,
            "batch_size": batch_size,
            "src_num_tokens": fseq_lengths.sum().item(),
            "tgt_num_tokens": sseq_lengths.sum().item() 
        }
        return export
    return _collate


def compute_tokenized_lengths(_file, tokenizer):
    save_path = '{}.lengths'.format(_file)
    if os.path.exists(save_path):
        print("{} exists, loading directly".format(save_path))
        _export = torch.load(save_path)
        return _export

    else:
        print("{} does not exist, computing".format(save_path))
        _export = torch.load(save_path)
        lines = open(_file).read().splitlines()
        _lens = []
        for line in lines:
            lang, tokens = tokenizer(line)
            _len = len(tokens)
            _lens.append(_len)

        N = len(lines)
        _lens = list(zip(range(N), _lens))
        _lens = sorted(_lens, key = lambda x: x[1])
        idxs, _lens = list(zip(*_lens))
        export = { "idxs": idxs, "lens": _lens}
        torch.save(export, save_path)
        print("{} computed and saved.".format(save_path))
        return export

class EpochBatchIterator:
    def __init__(self, dataset, collate_fn, max_tokens, shuffle=False):
        self.dataset = dataset
        self.collate = collate_fn
        self.max_tokens = max_tokens
        self._preconditions()
        self.shuffle = shuffle

    def _preconditions(self):
        def chunks(lengths):
            idx  = -1
            N = len(lengths)
            while idx < N - 1:
                total = 0
                start = idx + 1
                while total < self.max_tokens and idx < N-1:
                    idx += 1
                    total += lengths[idx]

                if total > self.max_tokens:
                    total -= lengths[idx]
                    idx -= 1

                yield (start, idx)

        lengths = self.dataset.export["lens"] 
        print("Obtaining batches")
        self.batches = list(chunks(lengths))
        self.toks = [sum(lengths[s:v]) for s, v in self.batches]
        print("Obtained {} batches".format(len(self.batches)))

    def __iter__(self):
        self.idx = -1
        if self.shuffle:
            random.shuffle(self.batches)
        return self

    def __next__(self):
        self.idx += 1
        if self.idx < len(self.batches):
            s, v = self.batches[self.idx]
            idxs = list(range(s, v))
            samples = [self.dataset[idx] for idx in idxs]
            return self.collate(samples)
        else:
            raise StopIteration

    def __len__(self):
        return len(self.batches)
