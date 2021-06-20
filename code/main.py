from configs import *
from utils import *
from environment import *
from model import *
from dataset import *
import numpy as np
import tqdm
from torch.nn import CrossEntropyLoss

myDataloader = MyDataloader()
train_loader, test_loader = myDataloader.dataloader()

model = Model(node_hidden_dim, edge_hidden_dim, gcn_num_layers, k).to(device)
criterion = CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
train_loss = []

for i in range(num_epochs):
    loss_per_epoch = 0
    for item in tqdm.tqdm(train_loader):
        graph, demand, distance = item[0].to(device), item[1].to(device), item[2].to(device)
        env = Environment(graph, demand, distance)
        sample_logprob, sample_distance, greedy_distance, target_matrix, predict_matrix = model(env)
        predict_matrix = predict_matrix.view(-1, 2)
        target_matrix = target_matrix.view(-1)
        classification_loss = criterion(predict_matrix.to(device), target_matrix.to(device))
        advantage = (sample_distance - greedy_distance).detach()
        reinforce = advantage * sample_logprob
        sequancial_loss = reinforce.sum()
        loss = alpha * sequancial_loss + beta * classification_loss
        loss.backward()
        optimizer.step()
        loss_per_epoch += loss
        break

    train_loss.append(loss_per_epoch)
    print('-train loss: %.4f' %train_loss[-1])
    break

torch.save(model.state_dict(), '../result/params.pkl')
write_loss('train_loss.txt', train_loss)