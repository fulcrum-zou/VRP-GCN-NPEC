from configs import *
from utils import *
from environment import *
from model import *
from dataset import *
import numpy as np
import tqdm
from torch.nn import CrossEntropyLoss

def train():
    loss_per_epoch = 0
    dist_per_epoch = 0
    batch_num = 0
    mean_dist_sample = 0
    mean_dist_greedy = 0
    for item in tqdm.tqdm(train_loader, 'train'):
        batch_num += 1
        graph, demand, distance = item[0].to(device), item[1].to(device), item[2].to(device)
        env = Environment(graph, demand, distance)
        sample_logprob, sample_distance, greedy_distance, target_matrix, predict_matrix = model(env)
        predict_matrix = predict_matrix.view(-1, 2)
        target_matrix = target_matrix.view(-1)
        classification_loss = criterion(predict_matrix.to(device), target_matrix.to(device))
        advantage = (sample_distance - greedy_distance).detach()
        mean_dist_sample += torch.mean(sample_distance)
        mean_dist_greedy += torch.mean(greedy_distance)
        reinforce = advantage * sample_logprob
        sequancial_loss = reinforce.sum()
        loss = alpha * sequancial_loss + beta * classification_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_per_epoch += loss
        if batch_num == 5:
            break
    loss_per_epoch /= (batch_size * batch_num)
    dist_per_epoch = mean_dist_sample / batch_num

    print('sample: %.2f greedy: %.2f' %(mean_dist_sample / batch_num, mean_dist_greedy / batch_num))
    return loss_per_epoch, dist_per_epoch, (mean_dist_sample.sum() < mean_dist_greedy.sum())

def test():
    loss_per_epoch = 0
    dist_per_epoch = 0
    batch_num = 0
    mean_dist_sample = 0
    mean_dist_greedy = 0
    with torch.no_grad():
        for item in tqdm.tqdm(test_loader, 'test '):
            batch_num += 1
            graph, demand, distance = item[0].to(device), item[1].to(device), item[2].to(device)
            env = Environment(graph, demand, distance)
            sample_logprob, sample_distance, greedy_distance, target_matrix, predict_matrix = model(env)
            predict_matrix = predict_matrix.view(-1, 2)
            target_matrix = target_matrix.view(-1)
            classification_loss = criterion(predict_matrix.to(device), target_matrix.to(device))
            advantage = (sample_distance - greedy_distance).detach()
            mean_dist_sample += torch.mean(sample_distance)
            mean_dist_greedy += torch.mean(greedy_distance)
            reinforce = advantage * sample_logprob
            sequancial_loss = reinforce.sum()
            loss = alpha * sequancial_loss + beta * classification_loss
            loss_per_epoch += loss
            if batch_num == 5:
                break

        loss_per_epoch /= (batch_size * batch_num)
        dist_per_epoch = mean_dist_sample / batch_num
    return loss_per_epoch, dist_per_epoch


if __name__ == '__main__':
    myDataloader = MyDataloader()
    train_loader, test_loader = myDataloader.dataloader()

    model = Model(node_hidden_dim, edge_hidden_dim, gcn_num_layers, k).to(device)
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loss, test_loss = [], []
    train_dist, test_dist = [], []

    for i in range(num_epochs):
        if i == 5:
            break
        train_loss_per_epoch, train_dist_per_epoch, update = train()
        test_loss_per_epoch, test_dist_per_epoch = test()
        
        train_loss.append(train_loss_per_epoch)
        train_dist.append(train_dist_per_epoch)
        test_loss.append(test_loss_per_epoch)
        test_dist.append(test_dist_per_epoch)
        print('epoch: %d -train loss: %.2f -distance: %.2f -test loss: %.2f -distance: %.2f' %(i, train_loss[-1], train_dist[-1], test_loss[-1], test_dist[-1]))
        write_loss('train_loss.txt', i, train_loss[-1])
        write_distance('train_dist.txt', i, train_dist[-1])
        write_loss('test_loss.txt', i, test_loss[-1])
        write_distance('test_dist.txt', i, test_dist[-1])
        # torch.save(model.state_dict(), '../result/params.pkl')
        plot_loss(train_loss)
        plot_dist(train_dist)

        if update:
            print('update')
            model.sequencialDecoderGreedy.load_state_dict(model.sequencialDecoderSample.state_dict())