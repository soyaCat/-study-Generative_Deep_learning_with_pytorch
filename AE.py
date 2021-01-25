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

batch_size = 512
lr = (1e-5)*5
trainEpochs = 100
showPointEpochs = 1
testEpochs = 1

train_mode = True
load_model = False
save_model = True
date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
save_path = "./saved_AE_model/"+date_time+"/model/"
load_path = "./saved_AE_model/"+"20210124-23-33-26"+"/model/"

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
        
        self.d_fc0 = nn.Linear(1024,2)
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

    def forward(self, x):
        out = self.Encoder(x)
        out = out.view(batch_size, -1)
        encoder_out = self.d_fc0(out)
        decoder_input = self.d_fc1(encoder_out)
        decoder_input = torch.reshape(decoder_input, shape=[-1, 64, 4, 4]).to(self.device)
        decoder_out = self.Decoder(decoder_input)
        return decoder_out, encoder_out

class AE():
    def __init__(self, device):
        self.device = device
        self.batch_size = batch_size
        self.model = AutoEncoder_model(device, self.batch_size).to(self.device)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.load_model = load_model
        self.load_path = load_path
        if self.load_model == True:
            print("model loaded!")
            self.model.load_state_dict(torch.load(self.load_path+"state_dict_model.pt"))


    def train_model(self, train_loader):
        self.model.train()
        loss_list = []
        for image, label in train_loader:
            x = image.to(self.device)
            decoder_out, encoder_out= self.model.forward(x)

            loss = self.loss(decoder_out, x)
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss)

        return torch.mean(torch.stack(loss_list))

    def get_result(self, img):
        self.model.eval()
        with torch.no_grad():
            x = img.to(self.device)
            decoder_out, encoder_out = self.model.forward(x)
            return x[0], decoder_out[0]

    def get_result_with_point(self, img):
        self.model.eval()
        with torch.no_grad():
            x = img.to(self.device)
            decoder_out, encoder_out = self.model.forward(x)
            return encoder_out
            
    




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    AE = AE(device)
    image_process = image_process()

    for epoch in range(trainEpochs):
        loss = AE.train_model(train_loader)
        print("")
        print("epoch: ", epoch)
        print(loss)

    if save_model == True:
        os.makedirs(save_path)

    if save_model == True:
        torch.save(AE.model.state_dict(), save_path+"state_dict_model.pt")
        print("Model saved..")
    
    for epoch in range(showPointEpochs):
        axis_x_list = []
        axis_y_list = []
        number_list = []
        color_table = ['b','g','r','c','m','y','k','tan','brown','blueviolet']
        i = 0
        for index,[image, label] in enumerate(train_loader):
            i+=1
            encoder_out = AE.get_result_with_point(image)
            print(encoder_out.size())#torch.Size([1024, 2])
            print(label.size())#torch.Size([1024])
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
            img, gen_img = AE.get_result(image)
            img = image_process.image_postprocess(img.cpu()).data.numpy()

            plt.figure(figsize=(5,5))
            plt.imshow(img)
            plt.show()

            gen_img = image_process.image_postprocess(gen_img.cpu()).data.numpy()
            plt.figure(figsize=(5,5))
            plt.imshow(gen_img)
            plt.show()