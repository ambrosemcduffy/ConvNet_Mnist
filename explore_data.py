import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms

# How many samples per batch to load
batch_size = 64
# convert data to torch.FloatTensor
transforms = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])

train_data = datasets.MNIST(root='data',
                            train=True,
                            download=True,
                            transform=transforms)
trainloader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)


def explore_data():
    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    images, labels = next(iter(trainloader))
    img = images.reshape(images.shape[0], 28, 28)
    img = np.squeeze(img)
    for i in range(1, columns*rows+1):
        fig.add_subplot(rows, columns, i)
        plt.title(labels[i].numpy())
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(img[i], cmap='gray')


explore_data()
