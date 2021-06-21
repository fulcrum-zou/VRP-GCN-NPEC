from configs import *
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

class MyDataloader():
    def __init__(self):
        self.train = np.load(file_path + 'data/' + file_name + '-training.npz')
        self.test = np.load(file_path + 'data/' + file_name + '-testing.npz')

    def load_data(self, data):
        graph, demand, distance = (data[i] for i in data.files)
        dataset = TensorDataset(torch.FloatTensor(graph), torch.FloatTensor(demand), torch.FloatTensor(distance))
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        return dataloader

    def dataloader(self):
        train_loader = self.load_data(self.train)
        test_loader = self.load_data(self.test)
        return train_loader, test_loader