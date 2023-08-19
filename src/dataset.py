from torch.utils.data import Dataset
import torch

class NERDataset(Dataset):
    def __init__(self, dataset, device):
        self.device = device
        self.data, self.labels = self.get_dataset(dataset)

    def get_dataset(self, raw_data):
        data = list(raw_data['encoded_tokens'])
        labels = list(raw_data['encoded_labels'])
        return torch.LongTensor(data).to(self.device), torch.LongTensor(labels).to(self.device)

    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'label': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)