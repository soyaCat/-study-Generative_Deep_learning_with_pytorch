from __future__ import generators
from typing import Generator
import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.linear import Bilinear

import torchvision.datasets as dset 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from PIL import Image
import matplotlib.pyplot as plt
import os
import datetime
import time
from torchinfo import summary
from tqdm import tqdm

batch_size = 512
discri_lr = 0.0008
generator_lr = 0.0004
gen_factor = 1
trainEpochs = 10
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

class model_discriminator(nn.Module):
    def __init__(self, device):
        super(model_discriminator, self).__init__()
        self.device = device
        self.discriminator = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 5, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 5, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.discriminator_fc1 = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        input_size = x.size()
        discriminator_out = self.discriminator(x)
        discriminator_out = discriminator_out.view(input_size[0],-1)
        discriminator_out = self.discriminator_fc1(discriminator_out)

        return discriminator_out

class model_generator(nn.Module):
    def __init__(self, device, z_dims, batch_size):
        super(model_generator, self).__init__()
        self.device = device
        self.z_dims = z_dims
        self.batch_size = batch_size

        self.generator_fc1 = nn.Sequential(
            nn.Linear(100,3136),
            nn.BatchNorm1d(3136),
            nn.LeakyReLU()
        )

        self.generator = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor = 2),
            nn.Conv2d(64,128,5, stride=1, padding = 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.UpsamplingBilinear2d(scale_factor = 2),
            nn.Conv2d(128,64,5, stride=1, padding = 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64,64,5, stride=1, padding = 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64,1,5, stride=1, padding = 2),
            nn.Tanh()
        )
    def forward(self,x):
        generator_input = self.generator_fc1(x)
        generator_input = generator_input.view(self.batch_size, 64, 7, 7)
        generator_output = self.generator(generator_input)

        return generator_output


class GAN():
    def __init__(self, device):
        self.device = device
        self.z_dims = z_dims
        self.batch_size = batch_size
        self.gen_factor = gen_factor
        self.model_discri = model_discriminator(device).to(self.device)
        self.model_generator = model_generator(device, self.z_dims, self.batch_size).to(self.device)
        self.optimizer_discri = torch.optim.Adam(self.model_discri.parameters(), lr = discri_lr)
        self.optimizer_generator = torch.optim.Adam(self.model_generator.parameters(), lr = generator_lr)
        self.loss = nn.BCELoss()
        summary(self.model_discri, input_size = (self.batch_size, 1, 28, 28), device = 'cuda')
        print(" ")
        summary(self.model_generator, input_size = (batch_size, self.z_dims), device = 'cuda')

    def train_discriminaotr(self,img):
        '''
        self.model_generator.eval()
        self.model_discri.train()
        '''
        total_loss = 0
        for param in self.model_discri.parameters():
            param.requires_grad = True
        for param in self.model_generator.parameters():
            param.requires_grad = False
        valid = torch.from_numpy(np.ones((self.batch_size,1))).to(self.device)
        valid = valid.to(dtype = torch.float32)
        fake = torch.from_numpy(np.zeros((self.batch_size,1))).to(self.device)
        fake = fake.to(dtype = torch.float32)
        x = img.to(self.device)
        discri_out = self.model_discri.forward(x)
        loss = self.loss(discri_out, valid)
        loss.backward()
        total_loss+=loss.item()
        self.optimizer_discri.step()

        vector = torch.randn(x.size()[0], self.z_dims).to(self.device)
        generator_out = self.model_generator.forward(vector)
        discri_out = self.model_discri.forward(generator_out)
        loss = self.loss(discri_out, fake)
        loss.backward()
        total_loss+=loss.item()
        self.optimizer_discri.step()
        
        return total_loss/2.0

    def train_generator(self, img):
        '''
        self.model_generator.train()
        self.model_discri.eval()
        '''
        for param in self.model_discri.parameters():
            param.requires_grad = False
        for param in self.model_generator.parameters():
            param.requires_grad = True
        valid = torch.from_numpy(np.ones((self.batch_size,1))).to(self.device)
        valid = valid.to(dtype = torch.float32)
        x = img.to(self.device)
        vector = torch.randn(x.size()[0], self.z_dims).to(self.device)
        generator_out = self.model_generator.forward(vector)
        discri_out = self.model_discri.forward(generator_out)
        loss = self.loss(discri_out, valid)
        loss.backward()
        self.optimizer_generator.step()

        return loss.item()

    def train_models(self, train_loader):
        dis_loss_list = []
        gen_loss_list = []
        for i, image in enumerate(tqdm(train_loader)):
            dis_loss = self.train_discriminaotr(image)
            dis_loss_list.append(dis_loss)
            #mini_list = []
            for i in range(gen_factor):
                gen_loss = self.train_generator(image)
                gen_loss_list.append(gen_loss)
                #mini_list.append(round(gen_loss,4))
            #print(mini_list)

        return np.mean(dis_loss_list), np.mean(gen_loss_list)
        
    def get_result_accuracy(self, train_loader):
        self.model_discri.eval()
        self.model_generator.eval()
        with torch.no_grad():  
            for image in train_loader:
                x = image.to(self.device)
                discri_out = self.model_discri.forward(x)
                vector = torch.randn(x.size()[0], self.z_dims).to(self.device)
                Generator_out = self.model_generator.forward(vector)

    def get_result_gen_img(self):
        self.model_discri.eval()
        self.model_generator.eval()
        with torch.no_grad(): 
            vector = torch.randn(512, self.z_dims).to(self.device)
            generator_out = self.model_generator.forward(vector)
        return generator_out[0]




    



if __name__ == "__main__":
    image_process = image_process()
    camel_train = image_process.image_preprocess(camel_train)
    train_loader = torch.utils.data.DataLoader(camel_train, batch_size=batch_size, 
                                            shuffle=True, num_workers=2, drop_last=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    GAN = GAN(device)
    
    for epoch in range(trainEpochs):
        discri_loss, generator_loss = GAN.train_models(train_loader)
        print("discriminator_loss:", discri_loss)
        print("generator_loss:", generator_loss)
        img = GAN.get_result_gen_img()
        img = image_process.image_postprocess(img.cpu()).data.numpy()
        plt.figure(figsize=(5,5))
        plt.imshow(img)
        plt.show()

    if save_model == True:
        os.makedirs(save_path)

    if save_model == True:
        torch.save(GAN.model_discri.state_dict(), save_path+"state_dict_model_discri.pt")
        torch.save(GAN.model_generator.state_dict(), save_path+"state_dict_model_generator.pt")
        print("Model saved..")