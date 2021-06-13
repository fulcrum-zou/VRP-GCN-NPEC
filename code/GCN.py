from configs import *
import torch.nn as nn

class GCN(nn.Module()):
    def __init__(self,
                 node_input_dim,
                 edge_input_dim,
                 node_hidden_dim,
                 edge_hidden_dim,
                 gcn_num_layers,
                 k):
        super(GCN, self).__init__()

        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.gcn_num_layers = gcn_num_layers
        self.k = k
        
        self.W1 = nn.Linear(2, self.node_hidden_dim)      # node_W1
        self.W2 = nn.Linear(2, self.node_hidden_dim // 2) # node_W2
        self.W3 = nn.Linear(1, self.node_hidden_dim // 2) # node_W3
        self.W4 = nn.Linear(1, self.edge_hidden_dim // 2) # edge_W4
        self.W5 = nn.Linear(1, self.edge_hidden_dim // 2) # edge_W5
        
        self.node_embedding = nn.Linear(self.node_hidden_dim, self.node_hidden_dim, bias=False) # Eq5
        self.edge_embedding = nn.Linear(self.edge_hidden_dim, self.edge_hidden_dim, bias=False) # Eq6

        self.gcn_layers = nn.ModuleList([GCNLayer(self.node_hidden_dim) for i in range(self.gcn_num_layers)])
        
        self.relu = nn.ReLU()

    def forward(self, x_c, x_d, m):
        '''
        @param x_c: coordination(batch_size, node_num(N+1), 2)
        @param x_d: demand(batch_size, node_num(N+1), 1)
        @param m: (batch_size, node_num(N+1), node_num(N+1))
        '''
        

class GCNLayer(nn.Module()):
    def __init__(self, hidden_dim):
        super(GCNLayer, self).__init__()

        # node GCN layers
        self.W_node = nn.Linear(hidden_dim, hidden_dim)
        self.V_node_in = nn.Linear(hidden_dim, hidden_dim)
        self.V_node = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attn = AttentionEncoder(hidden_dim)
        self.relu = nn.ReLU()
        self.ln1_node = nn.LayerNorm(hidden_dim)
        self.ln2_node = nn.LayerNorm(hidden_dim)

        # edge GCN layers
        self.W_edge = nn.Linear(hidden_dim, hidden_dim)
        self.V_edge_in = nn.Linear(hidden_dim, hidden_dim)
        self.V_edge = nn.Linear(2 * hidden_dim, hidden_dim)
        self.W1_edge = nn.Linear(hidden_dim, hidden_dim)
        self.W2_edge = nn.Linear(hidden_dim, hidden_dim)
        self.W3_edge = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.ln1_edge = nn.LayerNorm(hidden_dim)
        self.ln2_edge = nn.LayerNorm(hidden_dim)

        self.hidden_dim = hidden_dim


    def forward(self, x, e, neighbor_index):
        '''
        @param x: (batch_size, node_num(N+1), node_hidden_dim)
        @param e: (batch_size, node_num(N+1), node_num(N+1), edge_hidden_dim)
        @param neighbor_index: (batch_size, k)
        '''
        # node embedding
        batch_size, node_num = x.size[0], x.size[1]
        node_hidden_dim = x.size[-1]
        t = x.unsqueeze(1).repeat(1, node_num, 1, 1)

        neighbor_index = neighbor_index.unsqueeze(3).repeat(1, 1, 1, node_hidden_dim)
        neighbor = t.gather(2, neighbor_index)
        neighbor = neighbor.view(batch_size, node_num, -1, node_hidden_dim)

        h_nb_node = self.ln1_node(x + self.relu(self.W_node(self.attn(x, neighbor, neighbor))))
        h_node = self.ln2_node(h_nb_node + self.relu(self.V_node(torch.cat([self.V_node_in(x), h_nb_node], dim=-1))))

        # edge embedding
        x_from = x.unsqueeze(2).repeat(1, 1, node_num, 1)
        x_to = x.unsqueeze(1).repeat(1, node_num, 1, 1)
        h_nb_edge = self.ln1_edge(e + self.relu(self.W_edge(self.W1_edge(e) + self.W2_edge(x_from) + self.W3_edge(x_to))))
        h_edge = self.ln2_edge(h_nb_edge + self.relu(self.V_edge(torch.cat[self.V_edge_in(e), h_nb_edge], dim=-1)))

        return h_node, h_edge