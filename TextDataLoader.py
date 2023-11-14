import torch

from torchtext.data.functional import to_map_style_dataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader

class TextDataLoader:
    def __init__(self, data, text_preprocessor, batch_size) -> None:
        self.text_preprocessor = text_preprocessor

        # Define the proportions for the three splits
        num_train = int(len(data) * 0.85)
        num_valid = int(len(data) * 0.1)
        num_test = len(data) - num_train - num_valid  # Remaining for testing

        # Split the dataset into training, validation, and test sets
        split_train_, split_valid_, split_test_ = random_split(
            data, [num_train, num_valid, num_test]
        )

        self.train = DataLoader(
            split_train_, batch_size=batch_size, shuffle=True, collate_fn=self.collate_batch
        )
        self.valid = DataLoader(
            split_valid_, batch_size=batch_size, shuffle=True, collate_fn=self.collate_batch
        )
        self.test= DataLoader(
            split_test_, batch_size=batch_size, shuffle=True, collate_fn=self.collate_batch
        )

        self.number_of_classes = len(set([label for (label, text) in data]))

    def collate_batch(self, batch):
        label_list, text_list, offsets = [], [], [0]
        
        for _label, _text in batch:
            label_list.append(self.text_preprocessor.label_pipeline(_label))
            processed_text = torch.tensor(self.text_preprocessor.text_pipeline(_text), dtype=torch.int64)
            
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)

        text_list = torch.cat(text_list)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return label_list.to(device), text_list.to(device), offsets.to(device)


