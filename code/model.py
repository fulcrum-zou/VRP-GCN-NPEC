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
    
    def forward(self, x_c, x_d, m):
        '''
        @param x_c: coordination (batch_size, node_num(N+1), 2)
        @param x_d: demand (batch_size, node_num(N+1))
        @param m: distance (batch_size, node_num(N+1), node_num(N+1))
        '''
        # GCN encoder
        h_node, h_edge = self.GCN(x_c, x_d, m)
        batch_size, node_num, node_hidden_dim = h_node.shape

        # sequencial decoder
        last_node = torch.zeros((batch_size, 1)).long()
        hidden = torch.zeros((2, batch_size, node_hidden_dim))
        mask = torch.zeros((batch_size, node_num), dtype=torch.bool)
        mask[:, 0] = True
        ind, probability, hidden = self.sequencialDecoder(h_node, last_node, hidden, mask)

        # classification decoder
        clf_out = self.classificationDecoder(h_edge)
