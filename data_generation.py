import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd
import numpy as np

autoencoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3)
)
autoencoder.load_state_dict(torch.load('log/net_param1.pkl'))

testset = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
test_loader = Data.DataLoader(testset, batch_size=1)

data = np.array([0, 0, 0])
# classification = []


for step, (feature, label) in enumerate(test_loader):
    if step == 3000:
        break

    x = feature.view(-1, 28*28)
    y = autoencoder(x).detach().numpy()
    data = np.vstack((data, y))
    # classification.append(label.numpy())

data = np.delete(data, 0, axis=0)
data = pd.DataFrame(data)
data.to_csv('log/dataset1.csv')

