from configs import *
from environment import *
from model import *
from dataset import *
import numpy as np

myDataloader = MyDataloader()
train_loader, test_loader = myDataloader.dataloader()

for i, item in enumerate(train_loader):
    graph, demand, distance = item[0].to(device), item[1].to(device), item[2].to(device)
    break

model = Model(node_hidden_dim, edge_hidden_dim, gcn_num_layers, decode_type, k)
model(graph, demand, distance)