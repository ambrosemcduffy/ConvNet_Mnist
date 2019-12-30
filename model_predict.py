import torch
import matplotlib.pyplot as plt
import numpy as np
import pyttsx3
from model import Network
from torchvision import datasets
import torchvision.transforms as transforms

batch_size = 16
criterion = torch.nn.NLLLoss()

transforms = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])

test_data = datasets.MNIST(root='data',
                           train=False,
                           download=True,
                           transform=transforms)

testloader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True)

images, labels = next(iter(testloader))
state_dict = torch.load('data/checkpoint.pth')
net = Network().cuda()
net.load_state_dict(state_dict)


def say_prediction(image, index):
    pred = predict(image, index)
    engine = pyttsx3.init()
    engine.say("I predict the number is {}".format(pred))
    engine.runAndWait()


def predict(images, index=0):
    out = net(images.cuda())
    ps = torch.exp(out)
    pred = ps.max(dim=1)[1]
    pred = pred[index].cpu().numpy()
    images = images.reshape(images.shape[0], 28, 28, 1)
    img = images[index]
    plt.imshow(np.squeeze(img.numpy()), cmap='gray')
    return pred

say_prediction(images, 4)
