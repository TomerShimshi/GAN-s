from msilib.schema import Class
import torch
import numpy as np
import os
import torchvision
import torch.nn as nn
import torch.optim  as optim
import torchvision.datasets as datasets
import  torch.functional as F
from torchvision import  transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
#import pytorch_lightning as pl


class Descriminator(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.disc = nn.Sequential( nn.Linear(in_feature,128),
          nn.LeakyReLU(0.1),
          nn.Linear(128,1),
          nn.Sigmoid(),
          )
    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, image_dim):
        super().__init__()
        self.gen= nn.Sequential(
            nn.Linear(z_dim,256),
            nn.LeakyReLU(0.1),
            nn.Linear(256,image_dim),
            nn.Tanh(),
            
        )

    def forward(self, x):
        return self.gen(x)

#hyperparam
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr =1e-4
z_dim = 64

image_dim = 28*28*1
batch_size =32
num_epochs = 100

disc= Descriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size,z_dim)).to(device)
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

datset = datasets.MNIST(root='dataset/',transform=transform,download=True)
loader = DataLoader(datset,batch_size=batch_size,shuffle=True)
opt_disc= optim.Adam(disc.parameters(),lr=lr)
opt_gen= optim.Adam(gen.parameters(),lr=lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/FAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/FAN_MNIST/real")
step =0

for epoch in range(num_epochs):
    for batch_idx, (real,_) in enumerate(loader):
        real = real.view(-1,784).to(device)
        batch_size_temp = real.shape[0]

        noise = torch.randn(batch_size_temp,z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real,torch.ones_like(disc_real)) 
        disc_fake = disc(fake.detach()).view(-1)
        lossD_fake = criterion(disc_fake,torch.zeros_like(disc_real)) 
        lossD= (lossD_fake + lossD_fake)/2
        disc.zero_grad()
        lossD.backward() # we can retain graph =True
        opt_disc.step()

        ## train the generator
        output = disc(fake).view(-1)
        lossG= criterion(output,torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

    if epoch% 5 == 0:
        print(f'Epoch = [{epoch}/{num_epochs}]')
        print(f'lossG = {lossG} , lossD ={lossD}')
        with torch.no_grad():
            fake = gen(fixed_noise).reshape(-1,1,28,28)
            data = real.reshape(-1,1,28,28)
            img_grid_fake = torchvision.utils.make_grid(fake,normalize=True)
            img_grid_real = torchvision.utils.make_grid(real,normalize=True)
            writer_fake.add_image('MNIST Fake Images',img_grid_fake,global_step=step)
            writer_real.add_image('MNIST Real Images.jpg',img_grid_real,global_step=step)
        step+=1

