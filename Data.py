import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from TextClassificationModel import TextClassificationModel
from StocksNews import CustomDataset  

BATCH_SIZE = 64

tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# Specify the path to your data
data_path = './Stocks and News/sentiment.dat'


# Create an instance of your CustomDataset
custom_dataset = CustomDataset(data_path)
test_dataset = custom_dataset[:int(len(custom_dataset)/5)]
custom_dataset = custom_dataset[int(len(custom_dataset)/5):]

vocab = build_vocab_from_iterator(yield_tokens(custom_dataset), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)

    text_list = torch.cat(text_list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return label_list.to(device), text_list.to(device), offsets.to(device)

# Split data for training and testing
custom_dataset = to_map_style_dataset(custom_dataset)
num_train = int(len(custom_dataset) * 0.95)

split_train_, split_valid_ = random_split(
    custom_dataset, [num_train, len(custom_dataset) - num_train]
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

number_of_classes = len(set([label for (label, text) in custom_dataset]))
vocab_size = len(vocab)
embedding_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextClassificationModel(vocab_size, embedding_size, number_of_classes).to(device)
data_loaders = (train_dataloader, valid_dataloader, test_dataloader)
