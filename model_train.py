import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from model import Network
# Number of subprocesses to use
num_workers = 0
# How many samples per batch to load
batch_size = 16
epochs = 50
# convert data to torch.FloatTensor
transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(10),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5], [0.5])])

train_data = datasets.MNIST(root='data',
                            train=True,
                            download=True,
                            transform=transforms)

test_data = datasets.MNIST(root='data',
                           train=False,
                           download=True,
                           transform=transforms)

path = 'data/'
net = Network().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = torch.nn.NLLLoss()

valid_size = 0.2

num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

trainloader = torch.utils.data.DataLoader(train_data,
                                          batch_size,
                                          sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size,
                                           sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size,
                                          shuffle=True)


def train():
    steps = 0
    losses = []
    losses_test = []
    val_loss_min = np.inf
    for e in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0
        for data, target in trainloader:
            steps += 1
            optimizer.zero_grad()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = net(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        for data, target in valid_loader:
            net.eval()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = net(data)
            loss_val = criterion(output, target)
            valid_loss = loss_val.item() * data.size(0)
        train_loss = train_loss/len(trainloader.dataset)
        val_loss = valid_loss/len(valid_loader.dataset)
        print("epoch: {}, train_loss {}, val_loss {}".format(e+1,
              train_loss,
              val_loss))
        if val_loss <= val_loss_min:
            torch.save(net.state_dict(), path+'checkpoint.pth')
            val_loss_min = val_loss
            print("saving out model")
    return losses, losses_test, output, net


def overall_accuracy():
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data, target in test_loader:
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        output = net(data)
        loss = criterion(output, target)
        test_loss += loss.item() * data.size(0)
        ps = torch.exp(output)
        _, pred = torch.max(ps, dim=1)
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    overall = np.sum(class_correct)/np.sum(class_total)
    test_loss = test_loss/len(test_loader.dataset)
    return test_loss, overall


def plot_loss():
    plt.title('Loss')
    plt.plot(loss, label='training_loss')
    plt.plot(loss_test, color='green', label='test_loss')
    plt.legend(loc="upper left")
    plt.show()


loss, loss_test, output, model = train()
