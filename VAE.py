import torch
import torch.nn as nn
import numpy

import torchvision.datasets as dset 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

batch_size = 256
lr = (1e-5)*5
trainEpochs = 1900
testEpochs = 10
totalEpochs = trainEpochs + testEpochs

train_mode = True
load_model = False

print_interval = 100

mnist_train = dset.MNIST("./", train = False, transform= transforms.ToTensor(),
                        target_transform=None, download=False)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size = batch_size, 
                                            shuffle = True, num_workers=2, drop_last= True)

class AutoEncoder_model(nn.Module):
    def __init__(self, device, batch_size):
        super(AutoEncoder_model, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.Encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        self.mu = nn.Linear(1024, 2)
        self.log_var = nn.Linear(1024, 2)
        self.epsilon = torch.randn(self.batch_size, 2, device=self.device)

        self.d_fc1 = nn.Linear(2,1024)
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward_all(self, x):
        out = self.Encoder(x)
        out = out.view(batch_size, -1)
        mu_out = self.mu(out)
        log_var_out = self.log_var(out)
        encoder_out = mu_out + torch.exp(log_var_out/2) * self.epsilon
        
        d_fc1 = self.d_fc1(encoder_out)
        reshape_for_decoder = torch.reshape(d_fc1, shape = [-1, 64, 4, 4]).to(self.device)
        decoder_out = self.Decoder(reshape_for_decoder)
        return decoder_out, mu_out, log_var_out

class VAE():
    def __init__(self, device):
        self.device = device
        self.batch_size = batch_size
        self.model = AutoEncoder_model(device, self.batch_size).to(self.device)
        self.r_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)

    def train_model(self, train_loader):
        self.model.train()
        for image, label in train_loader:
            x = image.to(self.device)
            y_, mu_out, log_var_out = self.model.forward_all(x)

            r_loss = self.r_loss(x, y_)
            kl_loss = -0.5*torch.sum(1+log_var_out - torch.square(mu_out)-torch.exp(log_var_out), axis = 1)
            print(kl_loss.size())
            print(r_loss.size())
            print(kl_loss)
            print(r_loss)
            kl_loss.backward()

            totalLoss = r_loss + kl_loss
            totalLoss.backward()
            self.optimizer.step()
            print("vc")


    def get_result(self, train_loader):
        self.model.eval()
        with torch.no_grad():
            for image, label in train_loader:
                x = image.to(self.device)
                output = self.model.forward_all(x)
    




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VAE = VAE(device)
    VAE.train_model(train_loader)