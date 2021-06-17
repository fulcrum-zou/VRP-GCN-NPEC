from configs import *
from attention import *
import torch
import torch.nn as nn

class SequencialDecoder(nn.Module):
    def __init__(self, hidden_dim, decode_type):
        super(SequencialDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.decode_type = decode_type

        self.softmax = nn.Softmax(dim=1)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2)
        self.tanh = nn.Tanh()
        self.h = nn.Linear(hidden_dim, 1)
        self.W = nn.Linear(2, 1)
        self.pointer = AttentionPointer(hidden_dim, use_tanh=True)

    def forward(self, x, last_node, hidden, mask):
        '''
        @param x: (batch_size, node_num, hidden_dim)
        @param last_node: (batch_size, 1)
        @param hidden: (2, batch_size, hidden_dim)
        @param mask: (batch_size, node_num)
        '''
        batch_size = x.size(0)
        batch_idx = torch.arange(start=0, end=batch_size).unsqueeze(1)
        last_x = x[batch_idx, last_node].permute(1, 0, 2)
        _, hidden = self.gru(last_x, hidden)
        z = hidden[-1]
        # Eq 15
        _, u = self.pointer(z, x.permute(1, 0, 2))
        # Eq 16
        u = u.masked_fill_(mask, -np.inf)
        probs = self.softmax(u)
        if self.decode_type == 'sample':
            ind = torch.multinomial(probs, num_samples=1)
        else:
            ind = torch.max(probs, dim=1)[1].unsqueeze(1)
        probability = probs[batch_idx, ind].squeeze(1)
        return ind, probability, hidden

class ClassificationDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super(ClassificationDecoder, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )
        self.softmax = nn.Softmax(-1)

    def forward(self, e):
        '''
        @param e: (batch_size, node_num, node_num, hidden_dim)
        '''
        a = self.MLP(e)
        a = a.squeeze(-1)
        out = self.softmax(a)
        return out