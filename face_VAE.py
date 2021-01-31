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
import torchsummary
from tqdm import tqdm

batch_size = 64
lr = 0.0005
trainEpochs = 0
generateEpochs = 30
testEpochs = 0
r_loss_factor = 10000
z_dims = 200

train_mode = True
load_model = True
save_model = False
date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
save_path = "./saved_face_VAE_model/"+date_time+"/model/"
load_path = "./saved_face_VAE_model/"+"20210131-18-18-11"+"/model/"

celebA_train = dset.CelebA("./", split='train', transform=transforms.ToTensor(),
                        target_transform=None, download=False)
train_loader = torch.utils.data.DataLoader(celebA_train, batch_size=batch_size, 
                                            shuffle=True, num_workers=2, drop_last=True)
celebA_test = dset.CelebA("./", split='test', transform=transforms.ToTensor(),
                        target_transform=None, download=False)
test_loader = torch.utils.data.DataLoader(celebA_test, batch_size=200, 
                                            shuffle=True, num_workers=2, drop_last=True)

class image_process():
    def image_postprocess(self, tensor):
        img = tensor.clamp(0,1)
        img = torch.transpose(img,0,1)
        img = torch.transpose(img,1,2)
        return img

class AutoEncoder_model(nn.Module):
    def __init__(self, device, z_dims):
        super(AutoEncoder_model, self).__init__()
        self.device = device
        self.z_dims = z_dims
        self.Encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
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
        
        self.mu = nn.Linear(41216, self.z_dims)
        self.log_var = nn.Linear(41216, self.z_dims)
        self.d_fc1 = nn.Sequential(
            nn.Linear(self.z_dims, 41216),
            nn.ReLU(),
        )

        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding = 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        out = self.Encoder(x)
        #print(out.size())
        out = out.view(out.size()[0], -1)
        #print(out.size())
        mu_out = self.mu(out)
        log_var_out = torch.exp(0.5*self.log_var(out))
        epsilon = torch.randn_like(log_var_out).to(self.device)
        encoder_out = mu_out + log_var_out * epsilon

        decoder_input = self.d_fc1(encoder_out)
        decoder_input = torch.reshape(decoder_input, shape=[-1,64, 28, 23]).to(self.device)
        decoder_out = self.Decoder(decoder_input)
        return decoder_out, encoder_out, mu_out, log_var_out

    def decode(self, x):
        decoder_input = self.d_fc1(x)
        decoder_input = torch.reshape(decoder_input, shape=[-1,64, 28, 23]).to(self.device)
        decoder_out = self.Decoder(decoder_input)
        return decoder_out


class VAE():
    def __init__(self, device):
        self.device = device
        self.batch_size = batch_size
        self.z_dims = z_dims
        self.model = AutoEncoder_model(device, z_dims).to(self.device)
        torchsummary.summary(self.model, input_size = (3,218,178), device = 'cuda')
        self.r_loss_factor = r_loss_factor
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.load_model = load_model
        self.load_path = load_path
        if self.load_model == True:
            print("model loaded!")
            self.model.load_state_dict(torch.load(self.load_path+"state_dict_model.pt"))

    def loss_func(self, decoder_out, image, mu_out, log_var_out):
            BCE = nn.functional.binary_cross_entropy(decoder_out, image, reduction='mean')
            KLD = -0.5 * torch.sum(1 + log_var_out - mu_out.pow(2) - log_var_out.exp())

            return self.r_loss_factor*BCE + KLD, BCE, KLD

    def train_model(self, train_loader):
        print("ok?")
        self.model.train()
        r_loss_list = []
        kl_loss_list = []
        total_loss_list = []
        for image, label in tqdm(train_loader):
            x = image.to(self.device)
            self.optimizer.zero_grad()
            decoder_out, encoder_out ,mu_out, log_var_out = self.model.forward(x)

            total_loss, BCE_loss, KLD_loss = self.loss_func(decoder_out, x, mu_out, log_var_out)
            total_loss.backward()
            self.optimizer.step()

            total_loss_list.append(total_loss.item())
            r_loss_list.append(BCE_loss.item())
            kl_loss_list.append(KLD_loss.item())

        return np.mean(total_loss_list), np.mean(r_loss_list), np.mean(kl_loss_list)

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

    def get_result_from_point(self, vector):
        self.model.eval()
        with torch.no_grad():
            x = vector.to(self.device)
            decoder_out = self.model.decode(x)
            return decoder_out[0]

class Generate_from_point():
    def __init__(self, VAE):
        self.VAE = VAE

    def get_label_and_vector_list(self, test_loader):
        label_list = []
        Vector_list = []
        print("\n", "get label_list and vector_list from model...")
        for index,[image, label] in enumerate(tqdm(test_loader)):
            x = image.to(self.VAE.device)
            label = label.data.numpy()
            for i in range(np.shape(label)[0]):
                label_list.append(label[i,:])
            encoder_out = self.VAE.get_result_with_point(image)
            encoder_out = encoder_out.cpu().data.numpy()
            for i in range(np.shape(encoder_out)[0]):
                Vector_list.append(encoder_out[i,:])

        return label_list, Vector_list

    def get_feature_vector(self, feature_num, label_list, Vector_list):
        '''
        0: 5_o_Clock_Shadow 
        1: Arched_Eyebrows 
        2: Attractive 
        3: Bags_Under_Eyes 
        4: Bald 
        5: Bangs 
        6: Big_Lips 
        7: Big_Nose 
        8: Black_Hair 
        9: Blond_Hair 
        10: Blurry 
        11: Brown_Hair 
        12: Bushy_Eyebrows 
        13: Chubby 
        14: Double_Chin 
        15: Eyeglasses 
        16: Goatee 
        17: Gray_Hair 
        18: Heavy_Makeup 
        19: High_Cheekbones 
        20: Male 
        21: Mouth_Slightly_Open 
        22: Mustache 
        23: Narrow_Eyes 
        24: No_Beard 
        25: Oval_Face 
        26: Pale_Skin 
        27: Pointy_Nose 
        28: Receding_Hairline 
        29: Rosy_Cheeks 
        30: Sideburns 
        31: Smiling 
        32: Straight_Hair 
        33: Wavy_Hair 
        34: Wearing_Earrings 
        35: Wearing_Hat 
        36: Wearing_Lipstick 
        37: Wearing_Necklace 
        38: Wearing_Necktie
        39: Young
        '''
        feature_list = []
        none_feature_list = []
        for index,label in enumerate(label_list):
            if(label[feature_num] == 1):
                feature_list.append(Vector_list[index])
            else:
                none_feature_list.append(Vector_list[index])
        
        feature_mean = np.mean(feature_list, axis = 0)
        none_feature_mean = np.mean(none_feature_list, axis = 0)
        result_vector = feature_mean - none_feature_mean

        return result_vector

    def get_random_img(self, test_loader, except_feature_num):
        finish_indexing = False
        target_img = 0
        for index,[image, label] in enumerate(test_loader):
            x = image.to(self.VAE.device)
            label = label.data.numpy()
            for i in range(np.shape(label)[0]):
                if(label[i][except_feature_num] == 0):
                    target_img = x[i]
                    encoder_out = self.VAE.get_result_with_point(x)
                    target_Vector = encoder_out.cpu().data.numpy()[i]
                    finish_indexing = True
                    break
            if (finish_indexing == True):
                break

        return target_img, target_Vector
                    





if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VAE = VAE(device)
    image_process = image_process()
    Generate_from_point = Generate_from_point(VAE)

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
    
    for epoch in range(generateEpochs):
        if(epoch == 0):
            label_list, Vector_list = Generate_from_point.get_label_and_vector_list(test_loader)
        feature_vector = Generate_from_point.get_feature_vector(39, label_list, Vector_list)
        print("selecting image from data sets....")
        img, target_Vector = Generate_from_point.get_random_img(test_loader, 39)
        for level in range(8):
            print("Vector adjustment level: ", level)
            target_Vector = target_Vector + level*feature_vector*0.2
            target_Vector_torch = torch.from_numpy(target_Vector)
            img = VAE.get_result_from_point(target_Vector_torch)
            img = image_process.image_postprocess(img.cpu()).data.numpy()
            plt.figure(figsize=(5,5))
            plt.imshow(img)
            plt.show()




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