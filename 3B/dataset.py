import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle

class TokenDataset(Dataset):
    def __init__(self, data, context_length, device):
        self.context_length = context_length
        self.device = device
        self.data = torch.from_numpy(data.astype(np.int64)).to(self.device)

    def __len__(self):
        return len(self.data)-self.context_length

    def __getitem__(self, idx):
        return self.data[idx:idx+self.context_length], self.data[idx+1:idx+1+self.context_length]
    
def getData(data_dir, model_config, train_config):
    test_data = np.memmap(os.path.join(data_dir,'val.bin'), dtype=np.uint16, mode='r')
    train_data = np.memmap(os.path.join(data_dir,'train.bin'), dtype=np.uint16, mode='r')
    train_dataset = TokenDataset(train_data,model_config.context_length,train_config.device)
    test_dataset = TokenDataset(test_data,model_config.context_length,train_config.device)
    train_dataloader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=train_config.batch_size, shuffle=False)
    return train_dataloader, test_dataloader

def getBatch(split: str, train_dataloader:DataLoader, test_dataloader:DataLoader):
    data = train_dataloader if split == 'train' else test_dataloader
    return next(iter(data))

def getVocabSize(data_dir:str) -> int:
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = 50304
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
    return meta_vocab_size