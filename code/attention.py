from configs import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionEncoder, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x, neighbor):
        '''
        @param x: (batch_size, node_num, hidden_dim)
        @param neighbor: (batch_size, node_num, k, hidden_dim)
        '''
        # scaled dot-product attention
        x = x.unsqueeze(2)
        neighbor = neighbor.permute(0, 1, 3, 2)
        attn_score = F.softmax(torch.matmul(x, neighbor) / np.sqrt(self.hidden_dim), dim=-1) # (batch_size, node_num, 1, k)
        weighted_neighbor = attn_score * neighbor
        
        # aggregation
        agg = x.squeeze(2) + torch.sum(weighted_neighbor, dim=-1)
        
        return agg

class AttentionPointer(nn.Module):
    def __init__(self, hidden_dim, use_tanh=True):
        super(AttentionPointer, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_tanh = use_tanh