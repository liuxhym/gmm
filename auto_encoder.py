import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data

epoch = 3
batch_size = 64
lr = 0.002

trainset = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
train_loader = Data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode


autoencoder = AutoEncoder()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
loss_func = nn.MSELoss()

for episode in range(epoch):
    for step, (feature, label) in enumerate(train_loader):
        if step == 500:
            break
        x = feature.view(-1, 28*28)
        y = feature.view(-1, 28*28)

        encode, decode = autoencoder(x)

        loss = loss_func(decode, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 60 == 0:
            print('loss is {}'.format(loss))

# torch.save(autoencoder.encoder.state_dict(), 'log/net_param1.pkl')
