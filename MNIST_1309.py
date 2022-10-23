
import pandas as pd
import torch
import torch.nn as nn
import copy
import torch.optim as optim
import numpy as np

from torchvision import datasets, transforms
from torch.autograd import Variable

import random
import os

from models.binarized_modules import  BinarizeLinear



batch_size = 256
test_batch_size= 1000

num_epochs = 1
prob_sa1 = 00 / 100
iters = 10
simulate_SA = False
calculate_switching_energy = False



class red_net_4(nn.Module):
    def __init__(self):
        super(red_net_4, self).__init__()
        self.infl_ratio=1
        self.fc1 = BinarizeLinear(784, 128*self.infl_ratio, bias=False)
        self.htanh1 = nn.Hardtanh()
        self.fc2 = BinarizeLinear(128*self.infl_ratio, 128*self.infl_ratio, bias=False)
        self.htanh2 = nn.Hardtanh()
        self.fc3 = BinarizeLinear(128*self.infl_ratio, 10, bias=False)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)

        # Keys are layer names to simulate as stuck at; value is number of weights
        # need to automate filling up of values
        self.stuck_at_layers= { 'fc1' : [128, 784],
                                'fc2' : [128, 128],
                                'fc3' : [10,  128] }

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        return self.logsoftmax(x)


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,transform=transforms.Compose([
                        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))   ]) ),
                        batch_size=batch_size, shuffle=True )
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))  ])  ),
                        batch_size=test_batch_size, shuffle=True)

def simulate_nn(
        num_epochs,
        prob_sa1,
        iters,
        simulate_SA,
        calculate_switching_energy,
        lr,
        log_interval
):
    def org_reset():
        for p in list(model.parameters()):
            if hasattr(p, 'org'):
                p.org.copy_(p.data.clamp_(-1, 1))
        for p in list(model.parameters()):
            if hasattr(p, 'org'):
                p.data.copy_(p.org)

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            if epoch % 100 == 0:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1

            optimizer.zero_grad()
            loss.backward()
            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.data.copy_(p.org)
            optimizer.step()
            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.org.copy_(p.data.clamp_(-1, 1))

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:


                data, target = Variable(data), Variable(target)
                output = model(data)
                test_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    model = red_net_4()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    #for calculation of switching energy
    num_switches= {'-1to1':[],'1to-1':[]}
    old_wts = {key : np.ones(tuple(model.stuck_at_layers[key])).astype(np.dtype('i4')) for key in model.stuck_at_layers.keys()}
    new_wts = {key : np.ones(tuple(model.stuck_at_layers[key])).astype(np.dtype('i4')) for key in model.stuck_at_layers.keys()}



    sa1 = {layername: np.random.rand(model.stuck_at_layers[layername][0],model.stuck_at_layers[layername][1]) < prob_sa1 for layername in model.stuck_at_layers.keys() }
    if simulate_SA:
        print(f'probability of device stuck at 1: {100 * prob_sa1}%')


    for epoch in range(1, num_epochs + 1):

        train(epoch)
        test()

        ### This should come before test
        if simulate_SA:
            temp = model.fc1.weight.detach().numpy()
            temp[sa1['fc1']]=1
            model.state_dict()["fc1.weight"][:] = torch.Tensor(temp)

            temp2 = model.fc2.weight.detach().numpy()
            temp2[sa1['fc2']]=1
            model.state_dict()["fc2.weight"][:] = torch.Tensor(temp2)

            temp3 = model.fc3.weight.detach().numpy()
            temp3[sa1['fc3']]=1
            model.state_dict()["fc3.weight"][:] = torch.Tensor(temp3)

        #print(model.parameters())
        torch.save(model.state_dict(),f'saved_models/1309/try/epoch_{epoch}.pth')
        state_dict = torch.load(f'saved_models/1309/try/epoch_{epoch}.pth')
        model.load_state_dict(state_dict)
        #org_reset()

        if calculate_switching_energy:
            new_wts = {
                'fc1': model.fc1.weight.detach().numpy().astype(np.dtype('i4')),
                'fc2': model.fc2.weight.detach().numpy().astype(np.dtype('i4')),
                'fc3': model.fc3.weight.detach().numpy().astype(np.dtype('i4'))
                }
            num_switches['-1to1'].append( np.count_nonzero(new_wts['fc1']-old_wts['fc1'] == 2) +
                                                np.count_nonzero(new_wts['fc2']-old_wts['fc2'] == 2) +
                                                np.count_nonzero(new_wts['fc3']-old_wts['fc3'] == 2))

            num_switches['1to-1'].append( np.count_nonzero(new_wts['fc1']-old_wts['fc1'] == -2) +
                                            np.count_nonzero(new_wts['fc2']-old_wts['fc2'] == -2) +
                                            np.count_nonzero(new_wts['fc3']-old_wts['fc3'] == -2))

            #print(new_wts['fc3']-old_wts['fc3'])
            #print(num_switches)


            old_wts.update({
                'fc1' : new_wts['fc1'].copy(),
                'fc2' : new_wts['fc2'].copy(),
                'fc3' : new_wts['fc3'].copy()
            })

    df = pd.DataFrame(num_switches)
    df.to_excel('switching.xlsx')










if __name__ == '__main__':
    arguments = {
        'num_epochs': 1,
        'prob_sa1': 0 / 100,
        'iters' : 1,
        #'simulate_SA' : True,
        'simulate_SA' : False,
        'calculate_switching_energy' : True,
        'lr':  0.01,
        'log_interval' :20
    }
    simulate_nn(**arguments)



