from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class pipeline:
    def __init__(self, dataset) -> None:
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = build_vocab_from_iterator(self.yield_tokens(dataset), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.text_pipeline = lambda x: self.vocab(self.tokenizer(x))
        self.label_pipeline = lambda x: int(x) - 1

    def yield_tokens(self, data_iter):
        for _, text in data_iter:
            yield self.tokenizer(text)