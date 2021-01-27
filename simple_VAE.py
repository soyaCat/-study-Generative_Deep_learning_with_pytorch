
  
from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')


'''
import torch
import torch.nn as nn
import numpy as np

import torchvision.datasets as dset 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from PIL import Image
import matplotlib.pyplot as plt
import os
import datetime
import time

batch_size = 32
lr = 0.0005
trainEpochs = 200
showPointEpochs = 1
testEpochs = 1
r_loss_factor = 100

train_mode = True
load_model = False
save_model = True
date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
save_path = "./saved_VAE2_model/"+date_time+"/model/"
load_path = "./saved_VAE2_model/"+"20210124-23-33-26"+"/model/"

mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(),
                        target_transform=None, download=False)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, 
                                            shuffle=True, num_workers=2, drop_last=True)

class image_process():
    def image_postprocess(self, tensor):
        img = tensor.clamp(0,1)
        img = torch.transpose(img,0,1)
        img = torch.transpose(img,1,2)
        return img

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
        
        self.mu = nn.Linear(1024,2)
        self.log_var = nn.Linear(1024,2)
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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)

            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        out = self.Encoder(x)
        out = out.view(batch_size, -1)
        mu_out = self.mu(out)
        log_var_out = self.log_var(out)
        encoder_out = mu_out + torch.exp(log_var_out/2) * self.epsilon
        decoder_input = self.d_fc1(encoder_out)
        decoder_input = torch.reshape(decoder_input, shape=[-1, 64, 4, 4]).to(self.device)
        decoder_out = self.Decoder(decoder_input)
        return decoder_out, encoder_out, mu_out, log_var_out

class VAE():
    def __init__(self, device):
        self.device = device
        self.batch_size = batch_size
        self.model = AutoEncoder_model(device, self.batch_size).to(self.device)
        self.r_loss_factor = r_loss_factor
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.load_model = load_model
        self.load_path = load_path
        if self.load_model == True:
            print("model loaded!")
            self.model.load_state_dict(torch.load(self.load_path+"state_dict_model.pt"))


    def train_model(self, train_loader):
        self.model.train()
        r_loss_list = []
        kl_loss_list = []
        total_loss_list = []
        for image, label in train_loader:
            x = image.to(self.device)
            decoder_out, encoder_out ,mu_out, log_var_out = self.model.forward(x)

            r_loss = torch.mean(torch.square(x-decoder_out), axis = [1,2,3])
            r_loss = self.r_loss_factor*r_loss
            kl_loss = -0.5*torch.sum(1+log_var_out-torch.square(mu_out)-torch.exp(log_var_out), axis = 1)
            total_loss = r_loss+kl_loss
            total_loss = torch.mean(total_loss)
            total_loss.backward()
            self.optimizer.step()

            total_loss_list.append(total_loss)
            r_loss_list.append(r_loss)
            kl_loss_list.append(kl_loss)

        return torch.mean(torch.stack(total_loss_list)), torch.mean(torch.stack(r_loss_list)), torch.mean(torch.stack(kl_loss_list))

    def get_result(self, img):
        self.model.eval()
        with torch.no_grad():
            x = img.to(self.device)
            decoder_out, encoder_out , _, _ = self.model.forward(x)
            return x[0], decoder_out[0]

    def get_result_with_point(self, img):
        self.model.eval()
        with torch.no_grad():
            x = img.to(self.device)
            decoder_out, encoder_out , _, _ = self.model.forward(x)
            return encoder_out
            
    




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VAE = VAE(device)
    image_process = image_process()

    for epoch in range(trainEpochs):
        total_loss, r_loss, kl_loss = VAE.train_model(train_loader)
        print("")
        print("epoch: ", epoch)
        print(total_loss)
        print(r_loss)
        print(kl_loss)
    if save_model == True:
        os.makedirs(save_path)

    if save_model == True:
        torch.save(VAE.model.state_dict(), save_path+"state_dict_model.pt")
        print("Model saved..")
    
    for epoch in range(showPointEpochs):
        axis_x_list = []
        axis_y_list = []
        number_list = []
        color_table = ['b','g','r','c','m','y','k','tan','brown','blueviolet']
        i = 0
        for index,[image, label] in enumerate(train_loader):
            i+=1
            encoder_out = VAE.get_result_with_point(image)
            #print(encoder_out.size())#torch.Size([1024, 2])
            #print(label.size())#torch.Size([1024])
            print("stack points...")
            encoder_out = encoder_out.cpu().data.numpy()
            for index, number in enumerate(label):
                axis_x_list.append(encoder_out[index][0])
                axis_y_list.append(encoder_out[index][1])
                number_list.append(color_table[number])
            if i % 5 == 0:
                i = 0
                plt.scatter(axis_x_list, axis_y_list, c = number_list, s = 2)
                plt.show()
                axis_x_list = []
                axis_y_list = []
                number_list = []

    for epoch in range(testEpochs):
        for image, label in train_loader:
            img, gen_img = VAE.get_result(image)
            img = image_process.image_postprocess(img.cpu()).data.numpy()

            plt.figure(figsize=(5,5))
            plt.imshow(img)
            plt.show()

            gen_img = image_process.image_postprocess(gen_img.cpu()).data.numpy()
            plt.figure(figsize=(5,5))
            plt.imshow(gen_img)
            plt.show()

'''