
from __future__ import print_function
import copy
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
import switching_energy

# Training settings
if True:
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    #parser.add_argument('--epochs', type=int,      default=2,          metavar='N',help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--gpus', default=0,
                        help='gpus used for training - e.g 0,1,3')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()







num_epochs = 1
prob_sa1 = 00 / 100
iters = 10
simulate_SA = False
calculate_switching_energy = False








torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


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

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        return self.logsoftmax(x)




model = red_net_4()
model_used = copy.deepcopy(model)



if args.cuda:
    torch.cuda.set_device(3)
    #for i in model_dict: i.cuda()

criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=args.lr)
#optimizer = [optim.SGD(i.parameters(), lr=args.lr) for i in model_dict]
optimizer = optim.SGD(model_used.parameters(), lr=args.lr)


def train(epoch):
    model_used.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model_used(data)
        loss = criterion(output, target)

        #if epoch%100==0:
        #    optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

        optimizer.zero_grad()
        loss.backward()
        for p in list(model_used.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model_used.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(index):
    net_clone = copy.deepcopy(model_used)
    net_clone.train()
    net_clone.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = net_clone(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':

    for index in range(iters):

        ##################model = copy.deepcopy(model_used)
        # for input = 1, 1% of all devices are S.A.1
        # print('Enter percent of devices stuck at 1 ')
        # prob_sa1 = int(input())/100
        #contains database of what devices have stuck at faults
        sa1 = {'fc1': [],
               'fc2': [],
               'fc3': []
               }

        if simulate_SA:

            print(f'probability of device stuck at 1: {100 * prob_sa1}%')



            # Determine what  devices will stay SA1
            for i in range(len(model_used.fc1.weight)):
                sa1['fc1'].append([])
                for j in range(len(model_used.fc1.weight[i])):
                    sa1['fc1'][i].append(False if random.random()>prob_sa1 else True)
            for i in range(len(model_used.fc2.weight)):
                sa1['fc2'].append([])
                for j in range(len(model_used.fc2.weight[i])):
                    sa1['fc2'][i].append(False if random.random() > prob_sa1 else True)
            for i in range(len(model_used.fc3.weight)):
                sa1['fc3'].append([])
                for j in range(len(model_used.fc3.weight[i])):
                    sa1['fc3'][i].append(False if random.random() > prob_sa1 else True)


        for epoch in range(1, num_epochs + 1):


            train(epoch)

            if simulate_SA:
                for i in range(len(model_used.fc1.weight)):
                    for j in range(len(model_used.fc1.weight[i])):
                        with torch.no_grad():
                            if sa1['fc1'][i][j]:
                                #print(model_used.fc1.weight[i, j].tolist())
                                model_used.fc1.weight[i, j] = 1
                                #print(model_used.fc1.weight[i, j].tolist())

                for i in range(len(model_used.fc2.weight)):
                    for j in range(len(model_used.fc2.weight[i])):
                        with torch.no_grad():
                            if sa1['fc2'][i][j]:
                                model_used.fc2.weight[i, j] = 1
                for i in range(len(model_used.fc3.weight)):
                    for j in range(len(model_used.fc3.weight[i])):
                        with torch.no_grad():
                            if sa1['fc3'][i][j]:
                                model_used.fc3.weight[i, j] = 1




            test(index)

            if calculate_switching_energy:
                print(model_used.fc2.weight),
            if calculate_switching_energy:
                old_model= [copy.deepcopy(model_used.fc1.weight),
                            copy.deepcopy(model_used.fc2.weight),
                            copy.deepcopy(model_used.fc3.weight) ]
                print(old_model)


            #if epoch%10==0:
                #print('stop')

            #if epoch%100==0:
                #optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1
        print("#"*80)
        torch.save(model_used.state_dict(),f'saved_models/1309/try/{prob_sa1}_{index}.pth')

        model_used = copy.deepcopy(model)
        optimizer = optim.SGD(model_used.parameters(), lr=args.lr)