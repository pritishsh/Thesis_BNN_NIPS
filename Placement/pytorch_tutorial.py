import torch
from torchvision import datasets as dts
from torchvision.transforms import ToTensor
import torch.nn as nn




train_set = dts.MNIST(
    root='data',
    train=True,
    transform=ToTensor,
    download=True
)


test_set = dts.MNIST(
    root='data',
    train=False,
    transform=ToTensor,
    download=True
)


class try_nn(nn.Module):
    def __init__(self):
        super(try_nn, self).__init__()
        self.fc1 = nn.Linear(784, 128, bias=False)
        self.htanh1 = nn.Hardtanh()
        self.fc2 = nn.Linear(128, 128*self.infl_ratio, bias=False)
        self.htanh2 = nn.Hardtanh()
        self.fc3 = nn.Linear(128, 10, bias=False)
        self.logsoftmax=nn.LogSoftmax()



    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        return self.logsoftmax(x)


learning_rate = 0.1
model = try_nn()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_set):
        data, target = torch.tensor(data, requires_grad=True), torch.tensor(target, requires_grad=True)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        if epoch%100==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_set.dataset),
            100. * batch_idx / len(train_set), loss.item()))



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





