import os
from torch.utils.data import Dataset
import torch
import ilmulti as ilm
from collections import namedtuple
import random
from copy import deepcopy

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


