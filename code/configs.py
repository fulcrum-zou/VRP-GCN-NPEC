import torch

file_path = '../dataset/'
file_name = 'G-20'
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


node_hidden_dim = 100
edge_hidden_dim = 100
gcn_num_layers = 3

batch_size = 32
node_num = 20          # number of customers
initial_capacity = 40  # initial capacity of vehicles
k = 10         # number of nearest neighbors