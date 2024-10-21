from models.autoencoder.unet import *
import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, relu):
        super(AutoEncoder, self).__init__()
        self.relu = relu
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.repr_fw = UNet(in_dim, out_dim)
        self.repr_re = UNet(out_dim, in_dim)
        self.sigmoid = nn.Sigmoid()
        if relu:
            self.act = nn.ReLU()

    def forward(self, x):
        out = self.repr_fw(x)
        out = self.sigmoid(out) 
        if self.relu:
            recon = self.act(self.repr_re(out))
        # recon = self.act(self.repr_re(out))
        else:
            recon = self.repr_re(out)
        return out, recon