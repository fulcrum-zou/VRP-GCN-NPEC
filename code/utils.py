from configs import *
import numpy as np
import matplotlib.pyplot as plt
import re

def write_loss(file_name, epoch, loss):
    file_path = '../result/' + file_name
    mode = 'w' if epoch == 0 else 'a'
    f = open(file_path, mode)
    f.write('%d %.2f\n' %(epoch, loss))
    f.close()

def write_distance(file_name, epoch, dist):
    file_path = '../result/' + file_name
    mode = 'w' if epoch == 0 else 'a'
    f = open(file_path, mode)
    f.write('%d %.2f\n' %(epoch, dist))
    f.close()

def plot_loss(loss):
    plt.cla()
    file_path = '../result/' + 'loss.png'
    plt.plot(loss, color='skyblue', linewidth=1)
    plt.title('Train Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(file_path)
    
def plot_dist(dist):
    plt.cla()
    file_path = '../result/' + 'dist.png'
    plt.plot(dist, color='skyblue', linewidth=1)
    plt.title('Train Distance')
    plt.xlabel('epochs')
    plt.ylabel('distance')
    plt.savefig(file_path)

def write_file(file_name, train_result, test_result):
    file_path = '../result/' + file_name
    f = open(file_path, 'a')
    for i in range(len(train_result)):
        f.write('%.2f ' %train_result[i][0])
        f.write('%.2f\n' %train_result[i][1])
        f.write('%.2f ' %test_result[i][0])
        f.write('%.2f\n' %test_result[i][1])
    f.close()

def read_file(file_name):
    file_path = 'result/' + file_name
    train_result, test_result = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            temp = re.findall(r'-?\d+\.?\d*e?-?\d*?', line)
            if i % 2 == 0:
                train_result.append([float(temp[0]), float(temp[1])])
            else:
                test_result.append([float(temp[0]), float(temp[1])])
                
    f.close()
    return train_result, test_result