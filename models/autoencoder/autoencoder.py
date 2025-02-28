from models.autoencoder.unet import *
import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AutoEncoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.repr_fw = UNet(in_dim, out_dim)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        out = self.repr_fw(x)
        out = self.sigmoid(out) 
        return out