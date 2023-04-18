
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
import pandas as pd

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=250, metavar='N',
                    help='number of epochs to train (default: 100)')
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



class Net_default(nn.Module):
    def __init__(self):
        super(Net_default, self).__init__()
        self.infl_ratio=3
        self.fc1 = BinarizeLinear(784, 2048*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(2048*self.infl_ratio)
        self.fc2 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(2048*self.infl_ratio)
        self.fc3 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(2048*self.infl_ratio)
        self.fc4 = nn.Linear(2048*self.infl_ratio, 10)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        return self.logsoftmax(x)
class Net_no_bn(nn.Module):
    def __init__(self):
        super(Net_no_bn, self).__init__()
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
class red_net_1(nn.Module):
    def __init__(self):
        super(red_net_1, self).__init__()
        self.infl_ratio=1
        self.fc1 = BinarizeLinear(784, 512*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.fc3 = BinarizeLinear(512*self.infl_ratio, 512*self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.fc4 = nn.Linear(512*self.infl_ratio, 10)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        #x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc3(x)
        x = self.drop(x)
        #x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        return self.logsoftmax(x)
class red_net_2(nn.Module):
    def __init__(self):
        super(red_net_2, self).__init__()
        self.infl_ratio=1
        self.fc1 = BinarizeLinear(784, 128*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.fc3 = BinarizeLinear(128*self.infl_ratio, 128*self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.fc4 = nn.Linear(128*self.infl_ratio, 10)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        #x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc3(x)
        x = self.drop(x)
        #x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        return self.logsoftmax(x)
class red_net_3(nn.Module):
    def __init__(self):
        super(red_net_3, self).__init__()
        self.infl_ratio=1
        self.fc1 = BinarizeLinear(784, 128*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.fc2 = BinarizeLinear(128*self.infl_ratio, 128*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.fc3 = BinarizeLinear(128*self.infl_ratio, 10)
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
        self.stuck_at_layers = {'fc1': [128, 784],
                                'fc2': [128, 128],
                                'fc3': [10, 128]}

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        return self.logsoftmax(x)

#model = red_net_4()
#if args.cuda:
#    torch.cuda.set_device(3)
#    model.cuda()


#criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=args.lr)
#optimizer = optim.SGD(model.parameters(), lr=args.lr)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        #if epoch%100==0:
        #    optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

        optimizer.zero_grad()
        loss.backward()
        # this was original code:
        '''
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))
        '''
        #the next lines are modified
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp(-1.1))

        #if batch_idx % args.log_interval == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(data), len(train_loader.dataset),
        #        100. * batch_idx / len(train_loader), loss.item()))
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



if __name__ == '__main__':
    num_epochs  = 2000
    num_iters   = 1
    prob_sa1    = 0 / 100
    #iters       = 1
    learning_rate = 0.01
    simulate_SA = False
    calculate_switching_energy  =  True
    log_interval    = 20

    for iter in range(num_iters):
        model = red_net_4()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        #contains database of what devices have stuck at faults
        sa1 = {layername: np.random.rand(model.stuck_at_layers[layername][0],model.stuck_at_layers[layername][1]) < prob_sa1  for layername in model.stuck_at_layers.keys()}

        if simulate_SA:
            print(f'probability of device stuck at 1: {100 * prob_sa1}%')

        # for calculation of switching energy
        num_switches = {'-1to1': [], '1to-1': []}
        old_wts = {key: np.ones(tuple(model.stuck_at_layers[key])).astype(np.dtype('i4')) for key in
                   model.stuck_at_layers.keys()}
        new_wts = {key: np.ones(tuple(model.stuck_at_layers[key])).astype(np.dtype('i4')) for key in
                   model.stuck_at_layers.keys()}

        for epoch in range(1, num_epochs + 1):
            print("Epoch {}".format(epoch))
            train(epoch)

            #doing test() before making sa1 might do the trick
            test()

            #try saving at this point to check if stuck at actually hampers output
            #torch.save(model.state_dict(),f'saved_models/0910/iter_{iter}_epoch_{epoch}.pth')


            # this code is not working so its replaced by old iterative code
            #with torch.no_grad():
            #    model.fc1.weight[sa1['fc1']] = 1
            #    model.fc2.weight[sa1['fc2']] = 1
                #model.fc3.weight[sa1['fc3']] = 1

            #this is the old code:
            for i in range(len(model.fc1.weight)):
                for j in range(len(model.fc1.weight[i])):
                    with torch.no_grad():
                        if sa1['fc1'][i][j]:
                            # print(model.fc1.weight[i, j].tolist())
                            model.fc1.weight[i, j] = 1
                            model.fc1.weight.org[i, j] = 1
                            # print(model.fc1.weight[i, j].tolist())

            for i in range(len(model.fc2.weight)):
                for j in range(len(model.fc2.weight[i])):
                    with torch.no_grad():
                        if sa1['fc2'][i][j]:
                            model.fc2.weight[i, j] = 1
                            model.fc2.weight.org[i, j] = 1
            for i in range(len(model.fc3.weight)):
                for j in range(len(model.fc3.weight[i])):
                    with torch.no_grad():
                        if sa1['fc3'][i][j]:
                            model.fc3.weight[i, j] = 1
                            model.fc3.weight.org[i, j] = 1

            if calculate_switching_energy:
                new_wts = {
                    'fc1': model.fc1.weight.detach().numpy().astype(np.dtype('i4')),
                    'fc2': model.fc2.weight.detach().numpy().astype(np.dtype('i4')),
                    'fc3': model.fc3.weight.detach().numpy().astype(np.dtype('i4'))
                }
                num_switches['-1to1'].append(np.count_nonzero(new_wts['fc1'] - old_wts['fc1'] == 2) +
                                             np.count_nonzero(new_wts['fc2'] - old_wts['fc2'] == 2) +
                                             np.count_nonzero(new_wts['fc3'] - old_wts['fc3'] == 2))

                num_switches['1to-1'].append(np.count_nonzero(new_wts['fc1'] - old_wts['fc1'] == -2) +
                                             np.count_nonzero(new_wts['fc2'] - old_wts['fc2'] == -2) +
                                             np.count_nonzero(new_wts['fc3'] - old_wts['fc3'] == -2))

                # print(new_wts['fc3']-old_wts['fc3'])
                # print(num_switches)

                old_wts.update({
                    'fc1': new_wts['fc1'].copy(),
                    'fc2': new_wts['fc2'].copy(),
                    'fc3': new_wts['fc3'].copy()
                })

        df = pd.DataFrame(num_switches)
        df.to_excel('switching_2000.xlsx')

            # whenever I run test here, the model overrides manually set values
            # therefore test is commented
            #test()

            #print(model.parameters())

            #save at this point to store wts with simulation of SA1
            #torch.save(model.state_dict(),f'saved_models/0910/iter_{iter}_epoch_{epoch}.pth')

        # uncomment to save once per iteration after it has been saved for said number of epochs
        torch.save(model.state_dict(),f'saved_models/try/iter_{iter}_epoch_{epoch}.pth')

