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

import torch
from torchvision.transforms import ToTensor

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.infl_ratio=3
        self.fc1 = BinarizeLinear(784, 2048*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        #self.bn1 = nn.BatchNorm1d(2048*self.infl_ratio)
        self.fc2 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        #self.bn2 = nn.BatchNorm1d(2048*self.infl_ratio)
        self.fc3 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        #self.bn3 = nn.BatchNorm1d(2048*self.infl_ratio)
        self.fc4 = nn.Linear(2048*self.infl_ratio, 10)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        #x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        #x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        x = self.drop(x)
        #x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        return self.logsoftmax(x)

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

files_3px = ['3px/zero.png',
         '3px/one.png',
         '3px/two.png',
         '3px/three.png',
         '3px/four.png',
         '3px/five.png',
         '3px/six.png',
         '3px/seven.png',
         '3px/eight.png',
         '3px/nine.png',
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

if __name__ == "__main__":
    # load back the model
    feed_forward_net = Net()
    state_dict = torch.load("saved_models/no batch norm/epoch_13.pth")
    feed_forward_net.load_state_dict(state_dict)

    # load MNIST validation dataset
    _, validation_data = download_mnist_datasets()

    # get a sample from the validation dataset for inference
    x=0
    input, target = validation_data[x][0], validation_data[x][1]

    '''
    #print(feed_forward_net)
    print(feed_forward_net.fc2.weight)

    with torch.no_grad():
        feed_forward_net.fc2.weight[0,0]=-1

    
    for param in feed_forward_net.parameters():
        print(param.data)
    
    #print(feed_forward_net.fc2.weight [0][0].data)
    #feed_forward_net.fc2.weight [0][0].data = torch.tensor(-1)
    #print(feed_forward_net.fc2.weight [0][0].data)
    print(feed_forward_net.fc2.weight)
    '''

    for i in files_3px:
        print('-' * 200)
        print('Testing ' + i)
        img = np.invert(Image.open(i).convert('L')).ravel()
        # img = img.T
        img = img.reshape((1, 28, 28))/255
        img2 = torch.from_numpy(img).to(dtype=torch.float32)
        predicted2 = predict(feed_forward_net, img2, class_mapping)
        print(f"Prediction: {predicted2}", )

        #prediction = model.predict(img)
        # print(prediction)
    print('the end')
