from torch.utils.data import Dataset

class RedditDataSource(Dataset):
    def __init__(self, transform=None):
        self.data = self._load_data('./Stocks and News/' )
        self.labels = {1: "berish", 2: "bullish"}
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _load_data(self, data_path, other_model_predicion = None):
        dates = []
        with open(data_path + 'RedditNews.csv', 'r', encoding='utf-8') as file:
            lines = file.readlines()[1:]  # Skip header line
            news_dates = [''.join(line.split(',')[0]) for line in lines]
            news_data = [''.join(line.split(',')[1:]) for line in lines]
        with open(data_path + 'upload_DJIA_table.csv', 'r', encoding='utf-8') as file:
            lines = file.readlines()[1:]
            djia_dates = [''.join(line.split(',')[0]) for line in lines]
            djia_data = [float(''.join(line.split(',')[4])) - float(''.join(line.split(',')[1])) > 0 for line in lines]
        labels = []
        data = []
        for i, date in enumerate(news_dates):
            if date in djia_dates:
                labels.append(djia_data[djia_dates.index(date)] + 1)
                data.append(news_data[i])
        
        return list(zip(labels, data))