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

batch_size = 64
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