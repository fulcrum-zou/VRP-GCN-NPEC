from configs import *
import torch
import torch.nn as nn
import numpy as np

class AttentionEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionEncoder, self).__init__()
        self.hidden_dim = hidden_dim