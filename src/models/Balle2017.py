import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import numpy as np
import constriction
from training_utils import universal_train_model


class GDN(nn.Module):
    """Generalized Divisive Normalization layer.
    y_i = x_i / sqrt(beta_i + sum_j gamma_ij * x_j^2)
    """

    def __init__(
        self,
        channels,
        inverse=False,
        beta_min=1e-6,
        gamma_init=0.1,
        reparam_offset=2**-18,
    ):
        super().__init__()
        self.channels = channels
        self.inverse = inverse
        self.beta_min = beta_min
        self.reparam_offset = reparam_offset

        self.beta = nn.Parameter(torch.ones(channels))
        self.gamma = nn.Parameter(torch.eye(channels) * gamma_init)

    def _reparameterize(self, beta, gamma):
        # Ensure beta and gamma are positive as per the paper's suggestion (squaring/offset)
        # Here we use a simpler version: beta = sqrt(beta^2 + offset)
        # and gamma = sqrt(gamma^2 + offset)
        beta_p = torch.sqrt(beta**2 + self.reparam_offset)
        gamma_p = torch.sqrt(gamma**2 + self.reparam_offset)
        return beta_p, gamma_p

    def forward(self, x):
        beta, gamma = self._reparameterize(self.beta, self.gamma)

        # x shape: (B, C, H, W)
        x2 = x**2

        # Compute the normalization factor: beta + sum_j gamma_ij * x_j^2
        # This can be done via 1x1 convolution
        norm = F.conv2d(x2, gamma.view(self.channels, self.channels, 1, 1), bias=beta)

        if self.inverse:
            y = x * torch.sqrt(norm)
        else:
            y = x / torch.sqrt(norm)

        return y


class EntropyBottleneck(nn.Module):
    """Simple Entropy Bottleneck using additive uniform noise during training.
    Estimates rate using a learned non-parametric density model.
    """

    def __init__(self, channels, init_scale=10.0):
        super().__init__()
        self.channels = channels
        # For simplicity, we'll use a Gaussian model with learned scale per channel
        # In a full implementation, we'd use a non-parametric model (sum of sigmoids)
        self.log_scale = nn.Parameter(torch.zeros(channels))

    def forward(self, x, training=True):
        if training:
            # Add uniform noise [-0.5, 0.5]
            noise = torch.rand_like(x) - 0.5
            x_tilde = x + noise
        else:
            x_tilde = torch.round(x)

        # Calculate likelihood p(x_tilde)
        # Using a Gaussian as a proxy for the non-parametric model
        scale = torch.exp(self.log_scale).view(1, -1, 1, 1)

        # We want the probability mass of the bin [x-0.5, x+0.5]
        # Which is approximately the density p(x_tilde) since bin width is 1
        # p(x) = (1 / (sqrt(2pi) * scale)) * exp(-0.5 * (x/scale)^2)
        dist = torch.distributions.Normal(0, scale)
        likelihoods = dist.log_prob(x_tilde).exp()

        # Clip likelihoods for stability
        likelihoods = torch.clamp(likelihoods, min=1e-9)

        return x_tilde, likelihoods


class Balle2017(pl.LightningModule):
    """End-to-end Optimized Image Compression (Ballé et al. 2017).
    Architecture:
    Encoder: 3 stages of Conv (stride 4/2/2) + GDN
    Decoder: 3 stages of IGDN + Deconv (stride 2/2/4)
    """

    def __init__(self, channels=128, lambda_=0.01, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.name = "Balle2017"
        self.lambda_ = lambda_

        # Analysis transform (Encoder)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=9, stride=4, padding=4),
            GDN(channels),
            nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2),
            GDN(channels),
            nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2),
            GDN(channels),
        )

        # Entropy Bottleneck
        self.entropy_bottleneck = EntropyBottleneck(channels)

        # Synthesis transform (Decoder)
        self.decoder = nn.Sequential(
            GDN(channels, inverse=True),
            nn.ConvTranspose2d(
                channels, channels, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            GDN(channels, inverse=True),
            nn.ConvTranspose2d(
                channels, channels, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            GDN(channels, inverse=True),
            nn.ConvTranspose2d(
                channels, 3, kernel_size=9, stride=4, padding=4, output_padding=3
            ),
        )

    def training_step(self, batch, batch_idx):
        x = batch
        y = self.encoder(x)
        y_tilde, likelihoods = self.entropy_bottleneck(y, training=True)
        x_hat = self.decoder(y_tilde)

        # Distortion: MSE
        distortion = F.mse_loss(x_hat, x)

        # Rate: bits per pixel (bpp)
        # likelihoods shape: (B, C, H, W)
        # num_pixels: B * H_orig * W_orig
        # But we can just use the mean log likelihood over all elements and normalize by spatial reduction
        # Total bits = -sum(log2(likelihoods))
        rate = -torch.log2(likelihoods).mean()

        # Loss: D + lambda * R
        # Note: rate here is bits per latent element.
        # To get bpp, we'd multiply by (H_latent * W_latent * C) / (H_orig * W_orig)
        # But for optimization, any positive weight on rate works.
        loss = distortion + self.lambda_ * rate

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_dist", distortion)
        self.log("train_rate", rate)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        y = self.encoder(x)
        y_hat, likelihoods = self.entropy_bottleneck(y, training=False)
        x_hat = self.decoder(y_hat)

        distortion = F.mse_loss(x_hat, x)
        rate = -torch.log2(likelihoods).mean()
        loss = distortion + self.lambda_ * rate

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_dist", distortion)
        self.log("val_rate", rate)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def forward(self, x):
        y = self.encoder(x)
        y_hat, _ = self.entropy_bottleneck(y, training=False)
        x_hat = self.decoder(y_hat)
        return x_hat

    def forward_get_latent(self, x):
        y = self.encoder(x)
        y_hat, likelihoods = self.entropy_bottleneck(y, training=False)
        x_hat = self.decoder(y_hat)

        # Compress using constriction (ANS)
        # We need to provide a model for the ANS coder.
        # Since we used a Gaussian in training, we use QuantizedGaussian here.
        y_np = y_hat.cpu().numpy().astype(np.int32)
        scales = torch.exp(self.entropy_bottleneck.log_scale).cpu().detach().numpy()

        compressed_payloads = []
        for i in range(y_np.shape[0]):
            symbols = y_np[i].flatten()
            # Channel-wise scales
            c, h, w = y_np.shape[1:]
            chan_scales = np.repeat(scales, h * w)
            chan_means = np.zeros_like(chan_scales)

            # Use a reasonably large range for the quantized Gaussian
            model_family = constriction.stream.model.QuantizedGaussian(-1000, 1000)
            coder = constriction.stream.stack.AnsCoder()
            coder.encode_reverse(symbols, model_family, chan_means, chan_scales)
            compressed_payloads.append(coder.get_compressed().tobytes())

        # For simplicity return the first one or a concatenation
        return x_hat, b"".join(compressed_payloads)

    def evaluate_image(self, x):
        """Standardized evaluation method. Processes full image at once."""
        device = next(self.parameters()).device
        x = x.to(device)

        # Ensure dimensions are multiples of 16 (4 * 2 * 2 downsampling)
        C, H, W = x.shape
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16

        if pad_h > 0 or pad_w > 0:
            # Pad: (left, right, top, bottom)
            x_padded = F.pad(
                x.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect"
            ).squeeze(0)
        else:
            x_padded = x

        with torch.no_grad():
            # Process entire image
            recon_padded, payload = self.forward_get_latent(x_padded.unsqueeze(0))
            recon_padded = recon_padded.squeeze(0)

        # Unpad to original size
        full_reconstruction = recon_padded[:, :H, :W]

        return {"reconstruction": full_reconstruction, "compressed_payload": payload}


def train_model(datamodule, experiment_name, epochs, learning_rate, target_flops=None):
    return universal_train_model(
        Balle2017,
        datamodule,
        experiment_name,
        epochs,
        learning_rate,
        target_flops=target_flops,
    )
