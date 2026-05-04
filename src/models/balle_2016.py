import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dahuffman
import constriction

from .base import BaseAutoencoder

class GDN(nn.Module):
    """Generalized Divisive Normalization layer.
    From: "End-to-end Optimized Image Compression" (Ballé et al., 2016)
    """
    def __init__(self, in_channels, inverse=False, beta_min=1e-6, gamma_init=0.1):
        super().__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.in_channels = in_channels

        # Parameters beta and gamma as defined in the paper
        self.beta = nn.Parameter(torch.ones(in_channels))
        self.gamma = nn.Parameter(torch.eye(in_channels) * self.gamma_init)

    def forward(self, x):
        _, C, _, _ = x.size()
        
        # Enforce positivity on parameters (beta > 0, gamma > 0)
        # The paper uses a specific reparameterization, but clamping is a common robust alternative.
        beta = torch.clamp(self.beta, min=self.beta_min)
        gamma = torch.clamp(self.gamma, min=0.0)
        gamma = (gamma + gamma.t()) / 2  # Ensure symmetry as per paper
        
        # norm = sqrt(gamma * x^2 + beta)
        # Using 1x1 convolution for the weighted sum of squared channels
        norm = F.conv2d(x ** 2, gamma.view(C, C, 1, 1))
        norm = norm + beta.view(1, C, 1, 1)
        norm = torch.sqrt(norm)

        if self.inverse:
            return x * norm
        else:
            return x / norm

class EntropyBottleneck(nn.Module):
    """A simplified factorized entropy model.
    Models each channel as a Gaussian with learned mean and scale.
    While the original paper uses a non-parametric model, a learned parametric 
    model is the standard way to implement the 'Ballé 2016' logic in modern frameworks.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # Learned priors for each channel
        self.means = nn.Parameter(torch.zeros(channels))
        self.log_scales = nn.Parameter(torch.zeros(channels))

    def get_params(self):
        return self.means, torch.exp(self.log_scales)

    def forward(self, z):
        B, C, H, W = z.shape
        means, scales = self.get_params()
        
        # Reshape for broadcasting
        m = means.view(1, C, 1, 1)
        s = scales.view(1, C, 1, 1)

        if self.training:
            # Differentiable quantization simulation (Uniform Noise)
            noise = torch.rand_like(z) - 0.5
            z_hat = z + noise
        else:
            # True quantization
            z_hat = torch.round(z)

        # Estimate rate using log-likelihood: R = sum(-log2(P(z_hat)))
        # For training, we approximate the probability of the bin [z-0.5, z+0.5]
        # using the probability density function (PDF).
        # Standard normal PDF: exp(-0.5 * x^2) / sqrt(2*pi)
        # Log-likelihood of Gaussian: -0.5 * ((z_hat - m)/s)^2 - log(s) - 0.5 * log(2*pi)
        
        log_probs = -0.5 * ((z_hat - m) / s)**2 - self.log_scales.view(1, C, 1, 1) - 0.5 * np.log(2 * np.pi)
        
        # Convert nats to bits
        bits = torch.sum(-log_probs) / np.log(2.0)
        
        return z_hat, bits

class BalleEncoder(nn.Module):
    def __init__(self, N=128, M=128):
        super().__init__()
        # 4-stage architecture as described in the paper
        self.net = nn.Sequential(
            nn.Conv2d(3, N, 5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(N, N, 5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(N, N, 5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(N, M, 5, stride=2, padding=2)
        )

    def forward(self, x):
        return self.net(x)

class BalleDecoder(nn.Module):
    def __init__(self, N=128, M=128):
        super().__init__()
        # Mirrored architecture: Deconv -> IGDN
        self.net = nn.Sequential(
            nn.ConvTranspose2d(M, N, 5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, 3, 5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class Balle2016(BaseAutoencoder):
    def __init__(self, learning_rate=1e-4, N=128, M=128, rate_coeffitient=0.01):
        super().__init__(learning_rate=learning_rate)
        self.encoder = BalleEncoder(N=N, M=M)
        self.decoder = BalleDecoder(N=N, M=M)
        self.entropy_bottleneck = EntropyBottleneck(M)
        self.name = "Balle2016"
        self.rate_coeffitient = rate_coeffitient
        self.stride_requirement = 16
        
        # Compatibility with Pareto scripts
        self.quantization_bits = 8 

    def training_step(self, batch, batch_idx):
        x = batch
        x_padded, _ = self._pad(x)
        z = self.encoder(x_padded)
        z_hat, bits = self.entropy_bottleneck(z)
        x_hat_padded = self.decoder(z_hat)
        x_hat = self._unpad(x_hat_padded, x.shape[-2:])
        
        # Distortion (D)
        distortion = F.mse_loss(x_hat, x)
        
        # Rate (R) in bits per pixel
        # Normalizing by batch and spatial dimensions
        bpp = bits / (x.size(0) * x.size(2) * x.size(3))
        
        # Total Loss: L = D + lambda * R
        loss = distortion + self.rate_coeffitient * bpp
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("distortion", distortion, prog_bar=True)
        self.log("rate_bpp", bpp, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_padded, _ = self._pad(x)
        z = self.encoder(x_padded)
        z_hat, bits = self.entropy_bottleneck(z)
        x_hat_padded = self.decoder(z_hat)
        x_hat = self._unpad(x_hat_padded, x.shape[-2:])
        
        distortion = F.mse_loss(x_hat, x)
        bpp = bits / (x.size(0) * x.size(2) * x.size(3))
        loss = distortion + self.rate_coeffitient * bpp
        
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def forward(self, x):
        x_padded, orig_size = self._pad(x)
        z = self.encoder(x_padded)
        z_hat, _ = self.entropy_bottleneck(z)
        x_hat_padded = self.decoder(z_hat)
        return self._unpad(x_hat_padded, orig_size)

    def forward_just_cae(self, x):
        """Standard AE pass without entropy modeling."""
        x_padded, orig_size = self._pad(x)
        z = self.encoder(x_padded)
        x_hat_padded = self.decoder(z)
        return self._unpad(x_hat_padded, orig_size)

    def forward_get_latent(self, x):
        """Full Ballé 2016 pipeline using ANS coding (via constriction)."""
        x_padded, orig_size = self._pad(x)
        z = self.encoder(x_padded)
        B, C, H, W = z.shape
        
        # Quantize
        z_q = torch.round(z)
        
        # Reconstruct for evaluation
        x_hat_padded = self.decoder(z_q)
        x_hat = self._unpad(x_hat_padded, orig_size)
        
        # Coding using learned priors
        means, scales = self.entropy_bottleneck.get_params()
        
        # Prepare symbols and distributions for constriction
        symbols = z_q.detach().cpu().numpy().astype(np.int32).flatten()
        
        # Correctly broadcast means and scales to match flattened symbols (B, C, H, W)
        m_np = means.view(1, C, 1, 1).expand(B, C, H, W).detach().cpu().numpy().flatten()
        s_np = scales.view(1, C, 1, 1).expand(B, C, H, W).detach().cpu().numpy().flatten()
        
        # Use a QuantizedGaussian model for ANS coding
        try:
            model_family = constriction.stream.model.QuantizedGaussian(-1024, 1023)
            coder = constriction.stream.stack.AnsCoder()
            coder.encode_reverse(symbols, model_family, m_np, s_np)
            compressed_data = coder.get_compressed()
        except Exception as e:
            # Fallback to Huffman if ANS fails (e.g. range issues)
            codec = dahuffman.HuffmanCodec.from_data(symbols)
            compressed_data = codec.encode(symbols)
        
        return x_hat, compressed_data
