from configs import *
import numpy as np
import matplotlib.pyplot as plt
import re

def write_loss(file_name, loss):
    file_path = '../result/' + file_name
    f = open(file_path, 'w')
    for item in loss:
        f.write('%.4f\n' %item)
    f.close()

def write_file(file_name, train_result, test_result):
    file_path = '../result/' + file_name
    f = open(file_path, 'w')
    for i in range(len(train_result)):
        f.write('%.4f ' %train_result[i][0])
        f.write('%.4f\n' %train_result[i][1])
        f.write('%.4f ' %test_result[i][0])
        f.write('%.4f\n' %test_result[i][1])
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