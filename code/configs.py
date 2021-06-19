import torch

file_path = '../dataset/'
file_name = 'G-20'
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


node_hidden_dim = 100
edge_hidden_dim = 100
gcn_num_layers = 1

num_epochs = 10
batch_size = 64
beta = 1
learning_rate = 1e-4
weight_decay = 0.96

node_num = 20          # number of customers
initial_capacity = 5   # initial capacity of vehicles
k = 10                 # number of nearest neighbors
alpha = 1