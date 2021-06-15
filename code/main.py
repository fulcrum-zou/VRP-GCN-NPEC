from configs import *
from environment import *
from model import *
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data(file_name):
    train = np.load(file_path + 'data/' + file_name + '-training.npz')
    test = np.load(file_path + 'data/' + file_name + '-testing.npz')
    return train, test

def myDataLoader(graph, demand, distance):
    dataset = TensorDataset(torch.FloatTensor(graph), torch.FloatTensor(demand), torch.FloatTensor(distance))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataloader

train_data, test_data = load_data(file_name)
train_graph, train_demand, train_distance = (train_data[i] for i in train_data.files)
test_graph, test_demand, test_distance = [test_data[i] for i in test_data.files]

train_loader = myDataLoader(train_graph, train_demand, train_distance)
test_loader = myDataLoader(test_graph, test_demand, test_distance)

for i, item in enumerate(train_loader):
    graph, demand, distance = item[0].to(device), item[1].to(device), item[2].to(device)
    break

model = Model(node_hidden_dim, edge_hidden_dim, gcn_num_layers, decode_type, k)
model(graph, demand, distance)