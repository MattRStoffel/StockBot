import torch

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS as data
from torchtext.data.functional import to_map_style_dataset

from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader

from TextClassificationModel import TextClassificationModel

tokenizer = get_tokenizer("basic_english")



def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

train_iter, test_iter = data()

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        
        text_list.append(processed_text) # just a list of numbers, not actual words
        offsets.append(processed_text.size(0))
    
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)

    text_list = torch.cat(text_list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return label_list.to(device), text_list.to(device), offsets.to(device)

# Split data for training and testing
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)

BATCH_SIZE = 64  # batch size for training

split_train_, split_valid_ = random_split(
    train_dataset, [num_train, len(train_dataset) - num_train]
)
train_dataloader = DataLoader(
    split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
valid_dataloader = DataLoader(
    split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)

number_of_classes = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
embedding_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextClassificationModel(vocab_size, embedding_size, number_of_classes).to(device)
data_loaders = (train_dataloader, valid_dataloader, test_dataloader)
