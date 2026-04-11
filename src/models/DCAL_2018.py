# Deep Convolutional AutoEncoder-based Lossy Image Compression 2018
import torch
import torch.nn as nn

from .base import BaseAutoencoder


class DownBranch(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2

        def downsample_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size, stride=2, padding=padding),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.net = nn.Sequential(
            downsample_block(in_channels, out_channels),
            downsample_block(out_channels, out_channels),
            downsample_block(out_channels, out_channels),
            downsample_block(out_channels, out_channels)
        )

    def forward(self, x):
        return self.net(x)


class UpBranch(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2

        def upsample_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size, padding=padding),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(out_c, out_c * 4, kernel_size=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.net = nn.Sequential(
            upsample_block(in_channels, out_channels),
            upsample_block(out_channels, out_channels),
            upsample_block(out_channels, out_channels),
            upsample_block(out_channels, out_channels)
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.row1 = DownBranch(3, 42, kernel_size=3)
        self.row2 = DownBranch(3, 42, kernel_size=5)
        self.row3 = DownBranch(3, 44, kernel_size=7)

        self.pca_rotation = nn.Conv2d(128, 128, kernel_size=1, bias=False)

    def forward(self, x):
        out1 = self.row1(x)
        out2 = self.row2(x)
        out3 = self.row3(x)

        x_concat = torch.cat([out1, out2, out3], dim=1)
        return self.pca_rotation(x_concat)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.inverse_pca = nn.Conv2d(128, 128, kernel_size=1, bias=False)

        self.row1 = UpBranch(42, 42, kernel_size=3)
        self.row2 = UpBranch(42, 42, kernel_size=5)
        self.row3 = UpBranch(44, 44, kernel_size=7)

        self.final_merge = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.inverse_pca(x)

        x1, x2, x3 = torch.split(x, [42, 42, 44], dim=1)

        out1 = self.row1(x1)
        out2 = self.row2(x2)
        out3 = self.row3(x3)

        x_concat = torch.cat([out1, out2, out3], dim=1)
        return self.final_merge(x_concat)


class DCAL_2018(BaseAutoencoder):
    def __init__(self, learning_rate=1e-3):
        super().__init__(learning_rate=learning_rate)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)

        # quantizer
        # entropy coder

        x_hat = self.decoder(z)
        return x_hat
