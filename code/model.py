from configs import *
from GCN import *
from decoder import *
import torch.nn as nn

class Model(nn.Module):
    def __init__(self,
                 node_hidden_dim,
                 edge_hidden_dim,
                 gcn_num_layers,
                 k):
        super(Model, self).__init__()

        self.GCN = GCN(node_hidden_dim, edge_hidden_dim,
                       gcn_num_layers, k)
        self.sequencialDecoderSample = SequencialDecoder(node_hidden_dim, decode_type='sample', use_cuda=use_cuda)
        self.sequencialDecoderGreedy = SequencialDecoder(node_hidden_dim, decode_type='greedy', use_cuda=use_cuda)
        self.classificationDecoder = ClassificationDecoder(edge_hidden_dim)

    def seqDecoderForward(self, env, h_node, decode_type='sample'):
        # initialize last_node, hidden, mask & reset the environment
        env.reset()
        last_node = torch.zeros((batch_size, 1)).long().to(device)
        hidden = torch.zeros((2, batch_size, node_hidden_dim)).to(device)
        mask = torch.zeros((batch_size, node_num+1), dtype=torch.bool).to(device)
        mask[:, 0] = True
        log_prob = 0
        while env.all_visited() == False:
            # idx: (batch_size, 1)
            # prob: (batch_size)
            # hidden: (2, batch_size, hidden_dim)
            if decode_type=='sample':
                idx, prob, hidden = self.sequencialDecoderSample(h_node, last_node, hidden, mask)
            elif decode_type=='greedy':
                idx, prob, hidden = self.sequencialDecoderGreedy(h_node, last_node, hidden, mask)
            env.step(idx)
            last_node = idx
            log_prob = log_prob + torch.log(prob)
            mask = env.get_mask(idx)
        total_dist = env.calc_distance()
        matrix = env.decode_routes()

        return total_dist, log_prob, matrix
    
    def forward(self, env):
        x_c = env.graph
        x_d = env.demand
        m = env.distance

        # GCN encoder
        h_node, h_edge = self.GCN(x_c, x_d, m)
        batch_size, node_num, node_hidden_dim = h_node.shape

        # sequencial decoder
        # SampleRollout
        sample_distance, sample_logprob, target_matrix = self.seqDecoderForward(env, h_node, decode_type='sample')
        # print('sample:', env.routes[0])
        # GreedyRollout
        greedy_distance, _, _ = self.seqDecoderForward(env, h_node, decode_type='greedy')
        # print('greedy:', env.routes[0])

        # classification decoder
        predict_matrix = self.classificationDecoder(h_edge)

        return sample_logprob, sample_distance, greedy_distance, target_matrix, predict_matrix