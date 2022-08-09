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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from  Model_CNN import Discriminator,Generator,initialize_weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate= 2e-4
BATCH_SIZE =128
IMAGE_SIZE= 64
CHANNELS_IMAGE=3
Z_DIM=100
NUM_EPOCHES =50
FEAT_DISC =64
FEAT_GEN =64

transforms= transforms.Compose(
    [transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(CHANNELS_IMAGE)],[0.5 for _ in range(CHANNELS_IMAGE)])]
)

#dataset= datasets.MNIST(root = "dataset/", train=True,transform=transforms,download=True)
dataset = datasets.ImageFolder(root='celeb_dataset', transform=transforms)
loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
gen = Generator(Z_DIM,CHANNELS_IMAGE,FEAT_GEN).to(device)
disc = Discriminator(CHANNELS_IMAGE,FEAT_DISC).to(device)

initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(),lr=learning_rate,betas=(0.5,0.999))
opt_disc = optim.Adam(disc.parameters(),lr=learning_rate,betas=(0.5,0.999))
critirion = nn.BCELoss()

fixed_noise = torch.randn(32,Z_DIM,1,1).to(device)
writer_fake = SummaryWriter(f"logs/fake_celeb")
writer_real = SummaryWriter(f"logs/real_celeb")
step=0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHES):
    for batch_idx, (real,_) in enumerate(loader):
        real= real.to(device)
        noise= torch.randn((BATCH_SIZE,Z_DIM,1,1)).to(device)
        fake = gen(noise).to(device)

        #Train the descriminator
        disc_real = disc(real).reshape(-1)
        loss_disc_real= critirion(disc_real,torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        loss_dis_fake = critirion(disc_fake,torch.zeros_like(disc_fake))
        loss_disc= loss_dis_fake+loss_disc_real
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        #Train the generator
        output = disc(fake).reshape(-1)
        loss_gen = critirion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx% 100 == 0:
            print(f'Epoch = [{epoch}/{NUM_EPOCHES}]')
            print(f'lossG = {loss_gen} , lossD ={loss_disc}')
            with torch.no_grad():
                fake = gen(fixed_noise)
                #data = real.reshape(-1,1,28,28)
                img_grid_fake = torchvision.utils.make_grid(fake[:32],normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32],normalize=True)
                writer_fake.add_image('Fake',img_grid_fake,global_step=step)
                writer_real.add_image('Real',img_grid_real,global_step=step)
            step+=1
