import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import os
import numpy as np
import constriction


class GDN(nn.Module):
    """Generalized Divisive Normalization layer."""

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
        beta_p = torch.sqrt(beta**2 + self.reparam_offset)
        gamma_p = torch.sqrt(gamma**2 + self.reparam_offset)
        return beta_p, gamma_p

    def forward(self, x):
        beta, gamma = self._reparameterize(self.beta, self.gamma)
        x2 = x**2
        norm = F.conv2d(x2, gamma.view(self.channels, self.channels, 1, 1), bias=beta)
        if self.inverse:
            y = x * torch.sqrt(norm)
        else:
            y = x / torch.sqrt(norm)
        return y


class SELayer(nn.Module):
    """Squeeze-and-Excitation attention layer."""

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """Residual block with SE attention."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.se = SELayer(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.se(out)
        out += identity
        return out


class EntropyBottleneck(nn.Module):
    """Entropy Bottleneck using additive uniform noise during training."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.log_scale = nn.Parameter(torch.zeros(channels))

    def forward(self, x, training=True):
        if training:
            noise = torch.rand_like(x) - 0.5
            x_tilde = x + noise
        else:
            x_tilde = torch.round(x)

        scale = torch.exp(self.log_scale).view(1, -1, 1, 1)
        dist = torch.distributions.Normal(0, scale)
        likelihoods = dist.log_prob(x_tilde).exp()
        likelihoods = torch.clamp(likelihoods, min=1e-9)

        return x_tilde, likelihoods


class CustomCompressor(pl.LightningModule):
    """Custom Image Compression Model with Residual Blocks and Attention."""

    def __init__(self, channels=128, lambda_=0.01, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.name = "CustomCompressor"
        self.lambda_ = lambda_

        # Encoder: 4 stages of downsampling
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=5, stride=2, padding=2),
            GDN(channels),
            ResidualBlock(channels),
            nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2),
            GDN(channels),
            ResidualBlock(channels),
            nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2),
            GDN(channels),
            ResidualBlock(channels),
            nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2),
        )

        self.entropy_bottleneck = EntropyBottleneck(channels)

        # Decoder: 4 stages of upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                channels, channels, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            ResidualBlock(channels),
            GDN(channels, inverse=True),
            nn.ConvTranspose2d(
                channels, channels, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            ResidualBlock(channels),
            GDN(channels, inverse=True),
            nn.ConvTranspose2d(
                channels, channels, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            ResidualBlock(channels),
            GDN(channels, inverse=True),
            nn.ConvTranspose2d(
                channels, 3, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
        )

    def training_step(self, batch, batch_idx):
        x = batch
        y = self.encoder(x)
        y_tilde, likelihoods = self.entropy_bottleneck(y, training=True)
        x_hat = self.decoder(y_tilde)

        distortion = F.mse_loss(x_hat, x)
        rate = -torch.log2(likelihoods).mean()
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

        y_np = y_hat.cpu().numpy().astype(np.int32)
        scales = torch.exp(self.entropy_bottleneck.log_scale).cpu().detach().numpy()

        compressed_payloads = []
        for i in range(y_np.shape[0]):
            symbols = y_np[i].flatten()
            c, h, w = y_np.shape[1:]
            chan_scales = np.repeat(scales, h * w)
            chan_means = np.zeros_like(chan_scales)

            model_family = constriction.stream.model.QuantizedGaussian(-1000, 1000)
            coder = constriction.stream.stack.AnsCoder()
            coder.encode_reverse(symbols, model_family, chan_means, chan_scales)
            compressed_payloads.append(coder.get_compressed().tobytes())

        return x_hat, b"".join(compressed_payloads)

    def evaluate_image(self, x):
        device = next(self.parameters()).device
        x = x.to(device)

        # 4 stages of downsampling -> multiple of 16
        C, H, W = x.shape
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16

        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(
                x.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect"
            ).squeeze(0)
        else:
            x_padded = x

        with torch.no_grad():
            recon_padded, payload = self.forward_get_latent(x_padded.unsqueeze(0))
            recon_padded = recon_padded.squeeze(0)

        full_reconstruction = recon_padded[:, :H, :W]

        return {"reconstruction": full_reconstruction, "compressed_payload": payload}


def train_model(datamodule, experiment_name, epochs, learning_rate):
    model = CustomCompressor(learning_rate=learning_rate)
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

    best_model = CustomCompressor.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )

    print(
        f"Training complete. Best model saved to checkpoints/{os.path.basename(checkpoint_callback.best_model_path)}"
    )
    print("=" * 30)

    return best_model
