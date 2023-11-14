from torch.utils.data import Dataset

class sentiment_data(Dataset):
    def __init__(self, transform=None):
        self.data = self._load_data('./Stocks and News/sentiment.dat' )
        self.labels = {1: "berish", 2: "bullish"}
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
            data = [line.split('|')[2].join(line.split('|')[3:]) for line in lines]
            return list(zip(labels, data))