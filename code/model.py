from configs import *
from GCN import *
from decoder import *
import torch.nn as nn

class Model(nn.Module):
    def __init__(self,
                 node_hidden_dim,
                 edge_hidden_dim,
                 gcn_num_layers,
                 decode_type,
                 k):
        super(Model, self).__init__()

        self.GCN = GCN(node_hidden_dim, edge_hidden_dim,
                       gcn_num_layers, k)
        self.sequencialDecoder = SequencialDecoder(node_hidden_dim, decode_type)
        self.classificationDecoder = ClassificationDecoder(edge_hidden_dim)
    
    def forward(self, env):
        x_c = env.graph
        x_d = env.demand
        m = env.distance

        # GCN encoder
        h_node, h_edge = self.GCN(x_c, x_d, m)
        batch_size, node_num, node_hidden_dim = h_node.shape

        # sequencial decoder
        last_node = torch.zeros((batch_size, 1)).long()
        hidden = torch.zeros((2, batch_size, node_hidden_dim))
        mask = torch.zeros((batch_size, node_num), dtype=torch.bool)
        mask[:, 0] = True
        idx, probability, hidden = self.sequencialDecoder(h_node, last_node, hidden, mask)
        # idx: (batch_size, 1)
        # probability: (batch_size)
        # hidden: (2, batch_size, hidden_dim)

        # classification decoder
        clf_out = self.classificationDecoder(h_edge)
