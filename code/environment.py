from configs import *
from model import *

class Environment:
    def __init__(self, graph, demand, distance):
        self.graph = graph
        self.demand = demand
        self.distance = distance
        self.visited = np.zeros(graph.shape[1])
        
        self.batch_size = batch_size
        self.node_num = node_num
        self.initial_capacity = initial_capacity
        self.k_nearest = k_nearest

    def load_data(self):
        ''' load train dataset or test dataset
        '''

    def get_state(self):
        '''
        :return: customer coordinates and demand, vehicle states
        '''

    def step(self, action):
        ''' update customer and vehicle states
        :param action:
        :return: reward
        '''

    def mask(self):
        ''' compute the mask for current states
        '''

    def get_reward(self, state, action):
        '''
        :param state:
        :param action:
        :return: routing distance after the action
        '''