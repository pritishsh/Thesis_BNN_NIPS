# additional refernces:
# https://github.com/musikalkemist/pytorchforaudio/blob/main/03%20Making%20predictions/inference.py

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models.binarized_modules import  BinarizeLinear,BinarizeConv2d
from models.binarized_modules import  Binarize,HingeLoss
import numpy as np
from PIL import Image
import random

import torch
from torchvision.transforms import ToTensor

from main_mnist import Net_no_bn, Net_default, red_net_1, red_net_2, red_net_3, red_net_4

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1000, shuffle=True)

class_mapping = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]

files = ['zero.png',
         'one.png',
         'two.png',
         'three.png',
         'four.png',
         'five.png',
         'six.png',
         'seven.png',
         'eight.png',
         'nine.png',
         ]


def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    validation_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return train_data, validation_data

def predict(model, input, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        #expected = class_mapping[target]
    return predicted

def test(net_in):
    criterion = nn.CrossEntropyLoss()
    net_in.train()
    net_in.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:

            data, target = Variable(data), Variable(target)
            output = net_in(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))




if __name__ == "__main__":
    # load back the model
    state_dict = torch.load("saved_models/red_net_4/epoch_100.pth")
    feed_forward_net = red_net_4()
    feed_forward_net.load_state_dict(state_dict)

    print('original accuracy:')
    #test(feed_forward_net)


    #print(feed_forward_net)
    #test(feed_forward_net)

    #for input = 1, 1% of all devices are S.A.1
    #print('Enter percent of devices stuck at 1 ')
    #prob_sa1 = int(input())/100
    prob_sa1 = 1/10
    print(f'probability of device stuck at 1: {100*prob_sa1}%')

    sa1 = {'fc1': [],
           'fc2': [],
           'fc3': []
           }
    #print(feed_forward_net.fc1.weight)

    #fix what  devices will be SA1
    for i in range(len(feed_forward_net.fc1.weight)):
        sa1['fc1'].append([])
        for j in range(len(feed_forward_net.fc1.weight[i])):
            sa1['fc1'][i].append(False if random.random()>prob_sa1 else True)

    for i in range(len(feed_forward_net.fc2.weight)):
        sa1['fc2'].append([])
        for j in range(len(feed_forward_net.fc2.weight[i])):
            sa1['fc2'][i].append(False if random.random() > prob_sa1 else True)

    for i in range(len(feed_forward_net.fc3.weight)):
        sa1['fc3'].append([])
        for j in range(len(feed_forward_net.fc3.weight[i])):
            sa1['fc3'][i].append(False if random.random() > prob_sa1 else True)

    #print(feed_forward_net.fc1.weight)


    # automate this w.r.t arbitrary layer names
    # Fix wts for SA1 devices
    for i in range(len(feed_forward_net.fc1.weight)):
        for j in range(len(feed_forward_net.fc1.weight[i])):
            with torch.no_grad():
                if sa1['fc1'][i][j]:
                    feed_forward_net.fc1.weight[i, j] = 1

    for i in range(len(feed_forward_net.fc2.weight)):
        for j in range(len(feed_forward_net.fc2.weight[i])):
            with torch.no_grad():
                if sa1['fc2'][i][j]:
                    feed_forward_net.fc2.weight[i, j] = 1

    for i in range(len(feed_forward_net.fc3.weight)):
        for j in range(len(feed_forward_net.fc3.weight[i])):
            with torch.no_grad():
                if sa1['fc3'][i][j]:
                    feed_forward_net.fc3.weight[i, j] = 1

    '''
    #test(feed_forward_net)

    #print(feed_forward_net.fc2.weight)

    for i in range(len(feed_forward_net.fc2.weight)):
        for j in range(len(feed_forward_net.fc2.weight[i])):
            with torch.no_grad():
                if (i + j) % sa1_device_gap == 0:
                    feed_forward_net.fc2.weight[i, j] = 1
    #print(feed_forward_net.fc2.weight)

    #test(feed_forward_net)

    #print(feed_forward_net.fc3.weight)

    for i in range(len(feed_forward_net.fc3.weight)):
        for j in range(len(feed_forward_net.fc3.weight[i])):
            with torch.no_grad():
                if (i + j) % sa1_device_gap == 0:
                    feed_forward_net.fc3.weight[i, j] = 1
    #print(feed_forward_net.fc3.weight)
    '''
    print('accuracy with SA1')
    test(feed_forward_net)

















    '''
    for layer in feed_forward_net.children():
        #print(feed_forward_net[layer[0]])
        print(layer)
        print('----')
        if isinstance(layer, nn.Linear):
            print('###########')
            print(len(layer.state_dict()['weight']))
            print(len(layer.state_dict()['weight'][0]))
            print(layer.state_dict()['weight'])
    '''
    '''
    print(feed_forward_net.fc2.weight)

    with torch.no_grad():
        feed_forward_net.fc2.weight[0,0]=1

    print(feed_forward_net.fc2.weight)
    '''





