import ilmulti as ilm
import os
import torch
from collections import namedtuple

def persist(save_path, closure):
    if not os.path.exists(save_path):
        output = closure()
        torch.save(output, save_path)
    else:
        output = torch.load(save_path)
    return output

def preprocess(corpus, tokenizer, dictionary):
    def closure():
        lang_token = ilm.utils.language_token(corpus.lang)
        def _get(line):
            _lang, tokens = tokenizer(line)
            idxs = [dictionary.index(token) for token in tokens]
            return idxs

        content = open(corpus.path).read().splitlines()
        tensors = [_get(line) for line in content]
        return tensors
    save_path = '{}.processed.tensors'.format(corpus.path)
    return persist(save_path, closure)

def compute_tokenized_lengths(_file, tokenizer):
    Meta = namedtuple('Meta', 'idxs lens')
    def closure():
        lines = open(_file).read().splitlines()
        lengths = []
        for line in lines:
            lang, tokens = tokenizer(line)
            _len = len(tokens)
            lengths.append(_len)

        N = len(lines)
        lengths = list(zip(range(N), lengths))
        lengths = sorted(lengths, key = lambda x: x[1])
        idxs, lengths = list(zip(*lengths))
        #export = { "idxs": idxs, "lens": lengths}
        export = Meta(idxs, lengths)
        return export._asdict()
    save_path = '{}.processed.meta'.format(_file)
    loaded = persist(save_path, closure)
    return Meta(**loaded)



class CorpusHolding:
    def __init__(self, corpus, data, metadata):
        self.corpus = corpus
        self.data = data
        self.metadata = metadata

    @classmethod
    def read_corpus_in(cls, corpus, tokenizer, dictionary):
        data = preprocess(corpus, tokenizer, dictionary)
        metadata = compute_tokenized_lengths(corpus.path, tokenizer)
        return cls(corpus, data, metadata)


