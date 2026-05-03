import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_ae import BasicAE

class GDN(nn.Module):
    def __init__(self, in_channels, inverse=False, beta_min=1e-6, gamma_init=0.1):
        super().__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.in_channels = in_channels

        self.beta = nn.Parameter(torch.ones(in_channels))
        self.gamma = nn.Parameter(torch.eye(in_channels) * self.gamma_init)

    def forward(self, x):
        _, C, _, _ = x.size()
        
        # Enforce positivity
        beta = torch.clamp(self.beta, min=self.beta_min)
        gamma = torch.clamp(self.gamma, min=0.0)
        gamma = (gamma + gamma.t()) / 2  # Make symmetric
        
        # Apply normalization
        norm = F.conv2d(x ** 2, gamma.view(C, C, 1, 1))
        norm = norm + beta.view(1, C, 1, 1)
        norm = torch.sqrt(norm)

        if self.inverse:
            return x * norm
        else:
            return x / norm

class BalleEncoder(nn.Module):
    def __init__(self, M=192):
        super().__init__()
        self.conv1 = nn.Conv2d(3, M, kernel_size=5, stride=2, padding=2)
        self.gdn1 = GDN(M)
        self.conv2 = nn.Conv2d(M, M, kernel_size=5, stride=2, padding=2)
        self.gdn2 = GDN(M)
        self.conv3 = nn.Conv2d(M, M, kernel_size=5, stride=2, padding=2)
        self.gdn3 = GDN(M)
        self.conv4 = nn.Conv2d(M, M, kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.gdn3(self.conv3(x))
        x = self.conv4(x)
        return x

class BalleDecoder(nn.Module):
    def __init__(self, M=192):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(M, M, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.igdn1 = GDN(M, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(M, M, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.igdn2 = GDN(M, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(M, M, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.igdn3 = GDN(M, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(M, 3, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.igdn3(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        return x

class Balle2016(BasicAE):
    def __init__(self, learning_rate=1e-4, M=192):
        super().__init__(learning_rate=learning_rate)
        self.encoder = BalleEncoder(M=M)
        self.decoder = BalleDecoder(M=M)
        self.name = "Balle2016"
