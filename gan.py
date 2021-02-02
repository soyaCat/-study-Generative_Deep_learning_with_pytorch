from typing import Generator
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

batch_size = 256
lr = 0.004
trainEpochs = 20
showPointEpochs = 1
testEpochs = 1
r_loss_factor = 100
z_dims = 100

train_mode = True
load_model = False
save_model = True
date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
save_path = "./saved_gan_model/"+date_time+"/model/"
load_path = "./saved_gan_model/"+"20210124-23-33-26"+"/model/"

data_path = "./camel_data/camel_data.npy"
camel_train = np.load(data_path)

class image_process():
    def image_preprocess(self, tensor):
        result = (tensor-(255.0/2.0))/(255.0/2.0)
        result = torch.from_numpy(result)
        result = result.to(dtype = torch.float32)
        result = result.view(-1,1,28,28)
        return result

    def image_postprocess(self, tensor):
        img = tensor.clamp(0,1)
        img = torch.transpose(img,0,1)
        img = torch.transpose(img,1,2)
        return img

class model(nn.Module):
    def __init__(self, device, z_dims):
        super(model, self).__init__()
        self.device = device
        self.z_dims = z_dims
        self.discriminator = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.discriminator_fc1 = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

        self.generator_fc1 = nn.Sequential(
            nn.Linear(100,3136),
            nn.BatchNorm1d(3136),
            nn.LeakyReLU()
        )

        self.generator = nn.Sequential(
            nn
        )
    def forward(self, x):
        input_size = x.size()
        discriminator_out = self.discriminator(x)
        discriminator_out = discriminator_out.view(input_size[0],-1)
        discriminator_out = self.discriminator_fc1(discriminator_out)

        generator_input = torch.randn(input_size[0], self.z_dims).to(self.device)
        generator_input = self.generator_fc1(generator_input)
        generator_input = generator_input.view(input_size[0], 64, 7, 7)
        generator_output = self.generator(generator_input)


        return discriminator_out


class GAN():
    def __init__(self, device):
        self.device = device
        self.z_dims = z_dims
        self.batch_size = batch_size
        self.model = model(device, z_dims).to(self.device)
    
    def get_result_accuracy(self, train_loader):
        self.model.eval()
        with torch.no_grad():  
            for image in train_loader:
                x = image.to(self.device)
                discri_out = self.model.forward(x)
                print(discri_out)




    



if __name__ == "__main__":
    image_process = image_process()
    camel_train = image_process.image_preprocess(camel_train)
    train_loader = torch.utils.data.DataLoader(camel_train, batch_size=batch_size, 
                                            shuffle=True, num_workers=2, drop_last=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    GAN = GAN(device)
    GAN.get_result_accuracy(train_loader)