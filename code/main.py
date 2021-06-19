from configs import *
from environment import *
from model import *
from dataset import *
import numpy as np

myDataloader = MyDataloader()
train_loader, test_loader = myDataloader.dataloader()

model = Model(node_hidden_dim, edge_hidden_dim, gcn_num_layers, decode_type, k)

for i, item in enumerate(train_loader):
    graph, demand, distance = item[0].to(device), item[1].to(device), item[2].to(device)
    env = Environment(graph, demand, distance)
    # model(env)
    action = torch.full((batch_size, 1), 1)
    env.step(action)
    last_mask = torch.zeros(batch_size, node_num+1, dtype=torch.bool)
    env.mask(last_mask)
    break