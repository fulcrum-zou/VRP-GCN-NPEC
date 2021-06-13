from configs import *
from GCN import *
import torch.nn as nn

class Model(nn.Module):
    def __init__(self,
                 node_input_dim,
                 edge_input_dim,
                 node_hidden_dim,
                 edge_hidden_dim,
                 gcn_num_layers,
                 decode_type,
                 k):
        super(Model, self).__init__()

        self.GCN = GCN(node_input_dim, edge_input_dim,
                       node_hidden_dim, edge_hidden_dim,
                       gcn_num_layers, k)
        self.sequencialDecoder = SequencialDecoder(node_hidden_dim,
                                                   decode_type,
                                                   n_glimpses=0)
        self.classificationDecoder = ClassificationDecoder(edge_hidden_dim)
        