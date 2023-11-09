import torch
import re
import collections
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = self._load_data(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _load_data(self, data_path):

        with open(data_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()[1:]  # Skip header line
            labels = [int(line.split('|')[1]) + 1 for line in lines ] #if pattern.match(line)
            data = []
            words = ''
            for line in lines:
                words += line.split('|')[2].join(line.split('|')[3:])
            # if pattern.match(line):
                data.append(words)
                words = ''

            x = list(zip(labels, data))
        return x
