from configs import *
from model import *
from copy import deepcopy

class Environment:
    def __init__(self, graph, demand, distance):
        '''
        @param graph: (batch_size, node_num(N+1), 2)
        @param demand: (batch_size, node_num(N+1))
        @param distance: (batch_size, node_num(N+1), node_num(N+1))
        '''
        self.graph = graph
        self.demand = demand
        self.distance = distance
        
        self.batch_size = batch_size
        self.node_num = node_num
        self.initial_capacity = initial_capacity
        self.k = k

        self.visited, self.routes, self.remaining_capacity, self.remaining_demands = self.init_state()
        self.time_step = 0

    def init_state(self):
        '''
        visited: (batch_size, node_num+1)
        routes: (batch_size, 1)
        remaining_capacity: (batch_size, 1)
        remaining_demands: (batch_size, node_num+1)
        '''
        visited = torch.zeros(self.batch_size, self.node_num+1, dtype=torch.bool)
        visited[:, 0] = True
        routes = torch.full((self.batch_size, 1), 0)
        remaining_capacity = torch.full(size=(self.batch_size, 1), fill_value=self.initial_capacity, dtype=torch.float)
        remaining_demands = self.demand.clone().float()
        return visited, routes, remaining_capacity, remaining_demands

    def reset(self):
        self.visited, self.routes, self.remaining_capacity, self.remaining_demands = self.init_state()
        self.time_step = 0

    def step(self, action):
        ''' update customer and vehicle states
        @param action: (batch_size, idx(1))
        1. visited[idx] = True
        2. routes += action
        3. remaining_capacity = 
            if idx == 0: initial_capacity
            otherwise: max(0, remaining_capacity - demands[idx])
        4. remaining_demands[idx] = 0
        5. time_step += 1
        '''
        action = action.squeeze(-1)
        # 1.
        self.visited.scatter_(1, action.unsqueeze(1), True)
        # 2.
        self.routes = torch.cat((self.routes, action.unsqueeze(1)), dim=1)
        # 3.
        prev_capacity = self.remaining_capacity
        curr_demands = self.remaining_demands.gather(1, action.unsqueeze(1))
        self.remaining_capacity[action==0] = self.initial_capacity
        self.remaining_capacity[action!=0] = torch.maximum(torch.zeros(self.batch_size, 1), prev_capacity[action!=0] - curr_demands[action!=0])
        # 4.
        self.remaining_demands.scatter_(1, action.unsqueeze(1), 0)
        # 5.
        self.time_step = self.time_step + 1

    def get_mask(self, last_mask):
        ''' compute the mask for current states
        @param last_mask: (batch_size, node_num+1)
        1. if remaining_demands[idx] == 0 or
            remaining_demands[idx] >= remaining_capacity: set idx mask True
        2. if last_idx == 0 or t == 1: set the warehouse mask True
        3. if mask is all True: set the warehouse mask False
        '''
        mask = last_mask.clone()
        # 1.
        mask[self.remaining_demands==0] = True
        mask[self.remaining_demands>self.remaining_capacity] = True
        # 2.
        if self.time_step == 1:
            mask[:, 0] = True
        mask[self.routes[:, -2]==0, 0] = True
        # 3.
        mask[mask.all(dim=1), 0] = False
        return mask

    def dist_per_step(self, prev_step, curr_step):
        '''
        @param prev_step: (batch_size, 1)
        @param curr_step: (batch_size, 1)
        @return: distance of single step (batch_size, 1)
        '''
        idx = torch.arange(start=0, end=batch_size, step=1).unsqueeze(1)
        reward = self.distance[idx, prev_step, curr_step]
        return reward

    def get_reward(self):
        '''
        @return: routing distance after last action (batch_size, 1)
        '''
        prev_step = self.routes[:, -2:-1]
        curr_step = self.routes[:, -1:]
        reward = self.dist_per_step(prev_step, curr_step)
        return reward

    def calc_distance(self):
        '''
        @return: total distance of the routes (batch_size, 1)
        '''
        total_dist = torch.zeros(self.batch_size, 1)
        for i in range(1, self.routes.size(-1)):
            prev_step = self.routes[:, (i-1):i]
            curr_step = self.routes[:, i:(i+1)]
            dist = self.dist_per_step(prev_step, curr_step)
            total_dist = total_dist + dist
        return total_dist

    def decode_routes(self):
        ''' decode route sequence into a matrix
        @return: (batch_size, node_num+1, node_num+1)
        '''
        matrix = torch.zeros(self.batch_size, self.node_num+1, self.node_num+1, dtype=torch.float)
        idx = torch.arange(start=0, end=batch_size, step=1).unsqueeze(1)
        for i in range(1, self.routes.size(-1)):
            prev_step = self.routes[:, (i-1):i]
            curr_step = self.routes[:, i:(i+1)]
            matrix[idx, prev_step, curr_step] = 1

        return matrix.long()