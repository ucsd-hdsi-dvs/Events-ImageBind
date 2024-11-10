"""
Generative adversarial network implementation
"""
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from einops import rearrange
from models.resnet import ResNet

logger = logging.getLogger(__name__)

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(
            input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev,
                                  ndf * nf_mult,
                                  kernel_size=kw,
                                  stride=2,
                                  padding=padw,
                                  bias=use_bias)]
            sequence += [ nn.LeakyReLU(0.2, True) ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev,
                               ndf * nf_mult,
                               kernel_size=kw,
                               stride=1,
                               padding=padw,
                               bias=use_bias)]

        sequence += [ nn.LeakyReLU(0.2, True) ]
        sequence += [nn.Conv2d(ndf * nf_mult,
                               1,
                               kernel_size=kw,
                               stride=1,
                               padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


__all__ = ['GANLoss']

class Discriminator(nn.Module):
    def __init__(self, in_channels=3) -> None:
        super(Discriminator, self).__init__()
        self.resnet = ResNet(in_channels=in_channels, num_classes=2, layers=[1,2,2,1])

    def forward(self, x):
        out = self.resnet(x)
        return out

class GANLoss(nn.Module):
    def __init__(self, gan_k, in_channel=1, eps=1e-8, lr=1e-5, weight_decay=1e-5) -> None:
        super(GANLoss, self).__init__()
        self.gan_k = gan_k        
        self.discriminator = Discriminator(in_channels=in_channel)
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(), betas=(0, 0.9), eps=eps, lr=lr, weight_decay=weight_decay
        )

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.d_optimizer , 5, .5)

    def forward(self, fake, real):
        # Split real and fake images into separate channels and unsqueeze them to maintain batch dimension
        real = rearrange(real, 'b c h w -> (b c) 1 h w')
        fake = rearrange(fake, 'b c h w -> (b c) 1 h w')
        fake_detached = fake.detach()
        
        real_labels = torch.ones(real.shape[0], 2).to(fake.device)  # Real labels are 1
        fake_labels = torch.zeros(fake.shape[0], 2).to(fake.device)  # Fake labels are 0
        real_labels_for_fake = torch.ones(fake.shape[0], 2).to(fake.device)  # Real labels are 1
        

        self.discriminator.train()
        total_loss_d = 0

        for _ in range(self.gan_k):
            self.d_optimizer.zero_grad()
            # Calculate the discriminator losses for each fake and real image channel
            fake_out = self.discriminator(fake_detached)
            real_out = self.discriminator(real)
            losses_d = F.binary_cross_entropy_with_logits(fake_out, fake_labels) + F.binary_cross_entropy_with_logits(real_out, real_labels)
            losses_d.backward()
            self.d_optimizer.step()
            total_loss_d += losses_d.item()

        self.discriminator.eval()
        d_fake_probs = self.discriminator(fake)
        loss_g = F.binary_cross_entropy_with_logits(d_fake_probs, real_labels_for_fake)
        return loss_g, total_loss_d


class rgbGANLoss(nn.Module):
    def __init__(self, gan_k, in_channel=3, eps=1e-8, lr=1e-5, weight_decay=1e-5) -> None:
        super(rgbGANLoss, self).__init__()
        self.gan_k = gan_k        
        self.discriminator = Discriminator(in_channels=in_channel)
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(), betas=(0, 0.9), eps=eps, lr=lr, weight_decay=weight_decay
        )

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.d_optimizer , 5, .5)

    def forward(self, fake, real):
        fake_detached = fake.detach()
        
        real_labels = torch.ones(real.shape[0], 2).to(fake.device)  # Real labels are 1
        fake_labels = torch.zeros(fake.shape[0], 2).to(fake.device)  # Fake labels are 0
        real_labels_for_fake = torch.ones(fake.shape[0], 2).to(fake.device)  # Real labels are 1
        

        self.discriminator.train()
        total_loss_d = 0

        for _ in range(self.gan_k):
            self.d_optimizer.zero_grad()
            # Calculate the discriminator losses for each fake and real image channel
            fake_out = self.discriminator(fake_detached)
            real_out = self.discriminator(real)
            losses_d = F.binary_cross_entropy_with_logits(fake_out, fake_labels) + F.binary_cross_entropy_with_logits(real_out, real_labels)
            losses_d.backward()
            self.d_optimizer.step()
            total_loss_d += losses_d.item()

        self.discriminator.eval()
        d_fake_probs = self.discriminator(fake)
        loss_g = F.binary_cross_entropy_with_logits(d_fake_probs, real_labels_for_fake)
        return loss_g, total_loss_d


if __name__ == '__main__':
    test_fake = torch.randn([1, 3, 260, 346]).cuda()
    test_real = torch.rand_like(test_fake).cuda()
    # GAN test
    test_gan = rgbGANLoss(in_channel=3, gan_k=3).cuda()
    loss_g = test_gan(test_fake, test_real)
    print('generator loss: ', loss_g.item())
    