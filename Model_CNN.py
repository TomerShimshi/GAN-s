from audioop import bias
from cmath import tanh
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, image_chanels, feature_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
        #Input : N x image_chanels x 64 x64     
        nn.Conv2d(image_chanels, feature_d,kernel_size=4,stride=2,padding=1),#32 x 32
        nn.LeakyReLU(0.2),
        self._block(feature_d,feature_d*2,kernel_size=4,stride=2,padding=1), #16x16
        self._block(feature_d*2,feature_d*4,kernel_size=4,stride=2,padding=1), #8x8
        self._block(feature_d*4,feature_d*8,kernel_size=4,stride=2,padding=1),#4x4
        nn.Conv2d(feature_d*8,1,kernel_size=2,stride=2,padding=0), #1x1
        nn.Sigmoid(),
        )
    def _block(self,in_chanel,out_chanel,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(in_chanel, out_chanel, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_chanel),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        return self.disc(x)



class Generator(nn.Module):
    def __init__(self, z_dim, image_chanels,features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim,16*features_g,kernel_size=4,stride=1,padding=0), #N x f_g x 4x4x
            self._block(features_g*16,features_g*8,kernel_size=4,stride=2,padding=1), #8x8
            self._block(features_g*8,features_g*4,kernel_size=4,stride=2,padding=1), #16x16
            self._block(features_g*4,features_g*2 ,kernel_size=4,stride=2,padding=1),#32x32
            nn.ConvTranspose2d(features_g*2,image_chanels,kernel_size=4,stride=2,padding=1), #64x64
            nn.Tanh(),

        )
    
    def _block(self,in_chanel,out_chanel,kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_chanel, out_chanel, kernel_size=kernel_size, stride=stride,padding=padding,bias=False),
            nn.BatchNorm2d(out_chanel),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.gen(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data,0.0,0.02)

def test():
    N,in_chanels , H,W = 8,3,64,64
    z_dim =100
    x=torch.randn((N,in_chanels,H,W))
    disc = Discriminator(in_chanels,8)
    initialize_weights(disc)
    assert disc(x).shape == (N,1,1,1)

    gen= Generator(z_dim=z_dim,image_chanels=in_chanels,features_g=8)
    initialize_weights(gen)
    z= torch.randn((N,z_dim,1,1))
    assert gen(z).shape == (N,in_chanels,H,W)
    print('Successfull')

test()