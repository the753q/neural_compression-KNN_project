# Model based on
# Ballé et al., "Variational Image Compression with a Scale Hyperprior",
# ICLR 2018. https://arxiv.org/abs/1802.01436

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import numpy as np
import os
from compressai.layers import GDN
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
import pickle
import math
import struct

class AbsConv(nn.Sequential):
    def forward(self, x):
        return super().forward(x.abs())

class Hyperprior(pl.LightningModule):
    def __init__(self, lambda_=0.01, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.name = "Hyperprior"
        self.lambda_ = lambda_

        N = 64
        M = 96

        # Analysis transform (Encoder)
        self.g_a = nn.Sequential( 
            nn.Conv2d(3, N, kernel_size=5, stride=2, padding=2), 
            GDN(N), 
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2), 
            GDN(N), 
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2), 
            GDN(N), 
            nn.Conv2d(N, M, kernel_size=5, stride=2, padding=2), 
        )

        # Quantiztion and Entropy
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

        # Synthesis transform (Decoder)
        self.g_s = nn.Sequential(
            nn.ConvTranspose2d(M, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

        # Priors encoding
        self.h_a = AbsConv(
            nn.Conv2d(M, N, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
        )

        # Priors decoding
        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(N, M, kernel_size=3, stride=1, padding=1),
        )

    def training_step(self, batch, batch_idx):
        x = batch

        y = self.g_a(x)

        # hyperprior
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        sigma = self.h_s(z_hat)

        y_hat, y_likelihoods = self.gaussian_conditional(y, sigma)

        x_hat = self.g_s(y_hat)

        # Loss = D + lambda * R
        # Dividing by number of pixels makes it bbp, and independent of patch size
        B, C, H, W = x.shape
        num_pixels = B * H * W
        rate_y = -y_likelihoods.log2().sum() / num_pixels
        rate_z = -z_likelihoods.log2().sum() / num_pixels

        distortion = F.mse_loss(x_hat, x)

        loss = distortion + self.lambda_ * (rate_y + rate_z)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_dist", distortion)
        self.log("train_rate", rate_y + rate_z)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch

        y = self.g_a(x)

        # hyperprior
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        sigma = self.h_s(z_hat)

        y_hat, y_likelihoods = self.gaussian_conditional(y, sigma)

        x_hat = self.g_s(y_hat)

        # Loss = D + lambda * R
        # Dividing by number of pixels makes it independent of patch size
        # They are not true bpp of the payload, but the "potential"
        B, C, H, W = x.shape
        num_pixels = B * H * W
        rate_y = -y_likelihoods.log2().sum() / num_pixels
        rate_z = -z_likelihoods.log2().sum() / num_pixels

        distortion = F.mse_loss(x_hat, x)

        loss = distortion + self.lambda_ * (rate_y + rate_z)

        psnr = -10 * torch.log10(distortion)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_bpp_potential", rate_y + rate_z, prog_bar=True)
        self.log("val_psnr", psnr, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def forward(self, x):
        y = self.g_a(x)

        # hyperprior
        z = self.h_a(y)
        z_hat, _ = self.entropy_bottleneck(z)
        sigma = self.h_s(z_hat)

        y_hat, _ = self.gaussian_conditional(y, sigma)

        x_hat = self.g_s(y_hat)

        return x_hat
    
    def compress(self, x):
        # get latent
        y = self.g_a(x)

        # get prior
        z = self.h_a(y)
        
        # compress prior info
        z_strings, z_size = self.entropy_bottleneck.compress(z), z.shape[2:]
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.shape[2:])
        sigma = self.h_s(z_hat)
        # compress latent
        y_strings = self.gaussian_conditional.compress(y, sigma)

        # payload = pickle.dumps({
        #     "z_strings": z_strings,
        #     "z_size": z_size,
        #     "y_strings": y_strings
        # })

        # pack, big endian, u16x2 z dimensions, u32 len(z_strings), z_strings, y_strings
        payload = struct.pack(">HHI", z_size[0], z_size[1], len(z_strings[0])) + z_strings[0] + y_strings[0]

        return payload
    
    def decompress(self, payload):
        # data = pickle.loads(payload)

        # unpack, big endian, u16x2 z dimensions, u32 len(z_strings), 8 = 2+2+4
        z_h, z_w, z_len = struct.unpack(">HHI", payload[:8])
        data = {
            "z_strings": [payload[8:8+z_len]],
            "z_size": torch.Size([z_h, z_w]),
            "y_strings": [payload[8+z_len:]]
        }

        # decompress prior info
        z_hat = self.entropy_bottleneck.decompress(data["z_strings"], data["z_size"])

        sigma = self.h_s(z_hat)

        # decompress latent
        y_hat = self.gaussian_conditional.decompress(data["y_strings"], sigma)

        # reconstruct the image
        x_hat = self.g_s(y_hat)

        return x_hat

    def evaluate_image(self, x):
        """Standardized evaluation method. Processes full image at once."""
        # Build tables needed for encoding
        self.entropy_bottleneck.update()
        scale_table = torch.exp(torch.linspace(math.log(0.11), math.log(256), 64))
        self.gaussian_conditional.update_scale_table(scale_table)
        self.gaussian_conditional.update()

        device = next(self.parameters()).device
        x = x.to(device)

        # Ensure dimensions are multiples of 64
        C, H, W = x.shape
        pad_h = (64 - H % 64) % 64
        pad_w = (64 - W % 64) % 64

        if pad_h > 0 or pad_w > 0:
            # Pad: (left, right, top, bottom)
            x_padded = F.pad(
                x.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect"
            ).squeeze(0)
        else:
            x_padded = x

        with torch.no_grad():
            # Process entire image
            payload = self.compress(x_padded.unsqueeze(0))
            recon_padded = self.decompress(payload)
            recon_padded = recon_padded.squeeze(0)

        # Unpad to original size
        full_reconstruction = recon_padded[:, :H, :W]

        return {"reconstruction": full_reconstruction, "compressed_payload": payload}


def train_model(datamodule, experiment_name, epochs, learning_rate, lambda_):
    model = Hyperprior(learning_rate=learning_rate, lambda_=lambda_)

    checkpoint_filename = f"{experiment_name}-{model.name}-best"

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=checkpoint_filename,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    csv_logger = CSVLogger("logs/", name=experiment_name)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        precision="bf16-mixed",
        callbacks=[checkpoint_callback],
        logger=csv_logger,
    )

    print("=" * 30)
    print(f"Started experiment: {experiment_name}")
    print(f"Starting training for {model.name}...")
    trainer.fit(model, datamodule)

    best_model = Hyperprior.load_from_checkpoint(checkpoint_callback.best_model_path)

    print(
        f"Training complete. Best model saved to checkpoints/{os.path.basename(checkpoint_callback.best_model_path)}"
    )
    print("=" * 30)

    return best_model
