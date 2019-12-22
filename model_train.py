import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms

from model import Network
# Number of subprocesses to use
num_workers = 0
# How many samples per batch to load
batch_size = 64
epochs = 50
# convert data to torch.FloatTensor
transforms = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])

train_data = datasets.MNIST(root='data',
                            train=True,
                            download=True,
                            transform=transforms)

test_data = datasets.MNIST(root='data',
                           train=False,
                           download=True,
                           transform=transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True)

path = 'data/'
net = Network().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = torch.nn.NLLLoss()


def train():
    steps = 0
    print_every = 100
    losses = []
    losses_test = []
    for e in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            steps += 1
            optimizer.zero_grad()
            output = net(images.cuda())

            loss = criterion(output, labels.cuda())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            test_loss, acc = validation()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    print("Epochs:{}/{} train_loss: {} -- test_loss: {}\
                          -- accuracy: {}".format(e+1,
                          epochs,
                          running_loss/print_every,
                          test_loss,
                          acc))
                    running_loss = 0.0
        losses_test.append(test_loss)
        losses.append(running_loss)
        torch.save(model.state_dict(), path+'checkpoint.pth')
    return losses, losses_test, output, net


def validation():
    test_loss = 0.0
    acc = 0.0
    images, labels = next(iter(testloader))
    output = model(images.cuda())
    test_loss = criterion(output, labels.cuda()).item()
    ps = torch.exp(output)
    equality = labels.cpu().data == ps.cpu().max(dim=1)[1]
    acc = np.float32(equality.numpy()).mean()
    return test_loss, acc


def plot_loss():
    plt.title('Loss')
    plt.plot(loss, label='training_loss')
    plt.plot(loss_test, color='green', label='test_loss')
    plt.legend(loc="upper left")
    plt.show()


loss, loss_test, output, model = train()
