from ..preprocess import CorpusHolding

class CachedCorpusRegistry:
    def __init__(self):
        self._registry = {}

    def get(self, corpus, tokenizer, dictionary):
        key = corpus.__repr__()
        if not key in self._registry:
            self._registry[key] = CorpusHolding.read_corpus_in(
                    corpus, tokenizer, dictionary
            )
        return self._registry[key]

CORPUS_REGISTRY = CachedCorpusRegistry()
