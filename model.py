from torch import nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (3, 3))
        self.conv2 = nn.Conv2d(64, 128, (3, 3))
        self.conv3 = nn.Conv2d(128, 256, (3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2304, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 10)
        self.drop = nn.Dropout(.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
