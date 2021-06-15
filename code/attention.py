from configs import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

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

        self.project_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.project_x = nn.Conv1d(hidden_dim, hidden_dim, 1, 1)
        self.C = 10
        self.tanh = nn.Tanh()

        v = torch.FloatTensor(hidden_dim) 
        self.v = nn.Parameter(v)
        self.v.data.uniform_(-(1. / math.sqrt(hidden_dim)) , 1. / math.sqrt(hidden_dim))

    def forward(self, hidden, x):
        '''
        @param hidden: (batch_size, hidden_dim)
        @param x: (node_num, batch_size, hidden_dim)
        '''
        x = x.permute(1, 2, 0)
        q = self.project_hidden(hidden).unsqueeze(2)  # batch_size x hidden_dim x 1
        e = self.project_x(x)  # batch_size x hidden_dim x node_num 
        # expand the hidden by node_num
        # batch_size x hidden_dim x node_num
        expanded_q = q.repeat(1, 1, e.size(2)) 
        # batch x 1 x hidden_dim
        v_view = self.v.unsqueeze(0).expand(
                expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x node_num]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u  
        return e, logits