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
train_dist = []

for i in range(num_epochs):
    if i == 2:
        break
    loss_per_epoch = 0
    dist_per_epoch = 0
    batch_num = 0
    for item in tqdm.tqdm(train_loader):
        batch_num += 1
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
        dist_per_epoch += env.calc_distance().sum()
        if batch_num == 3:
            break
        
    loss_per_epoch /= (batch_size * batch_num)
    dist_per_epoch /= (batch_size * batch_num)
    train_loss.append(loss_per_epoch)
    train_dist.append(dist_per_epoch)
    print('epoch: %d -train loss: %.2f - distance: %.2f' %(i, train_loss[-1], train_dist[-1]))
    write_loss('train_loss.txt', i, train_loss[-1])
    write_distance('train_dist.txt', i, train_dist[-1])
    # torch.save(model.state_dict(), '../result/params.pkl')
    plot_loss(train_loss)
    plot_dist(train_dist)