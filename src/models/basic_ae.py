import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import constriction
import numpy as np
import dahuffman

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import os

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.prelu1 = nn.PReLU(16)
        self.prelu2 = nn.PReLU(32)
        self.prelu3 = nn.PReLU(64)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.prelu1 = nn.PReLU(32)
        self.prelu2 = nn.PReLU(16)

    def forward(self, x):
        x = self.prelu1(self.deconv1(x))
        x = self.prelu2(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x

class BasicAE(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.name = "BasicAE"
        self.quantization_bits = 8 # lower -> more compression
        self.rate_coeffitient = 1.0  # higher -> more compression
        assert self.rate_coeffitient >= 0.0
    
    def pass_to_encoders(self, x):
        encoded = self.encoder(x)
        return encoded

    def pass_to_decoders(self, x):
        decoded = self.decoder(x)
        return decoded

    def entropy_coder(self, x):
        symbols = x.cpu().numpy().astype(np.int32).flatten()
        # tile to match batch dimension in symbols
        means = np.tile(self.z_means.cpu().numpy().flatten(), x.shape[0]).astype(np.float64)
        stds = np.tile((self.z_stds + 1e-8).cpu().numpy().flatten(), x.shape[0]).astype(np.float64)

        B = self.quantization_bits
        model_family = constriction.stream.model.QuantizedGaussian(-2**(B-1), 2**(B-1)-1)
        coder = constriction.stream.stack.AnsCoder()
        coder.encode_reverse(symbols, model_family, means, stds)
        compressed =  coder.get_compressed()

        print(symbols[:20])
        print(f"before {len(symbols)} after {len(compressed)}")

        return compressed

    def entropy_decoder(self, x, original_shape):
        # tile to match batch size
        batch_size = original_shape[0]
        means = np.tile(self.z_means.cpu().numpy().flatten(), batch_size).astype(np.float64)
        stds = np.tile((self.z_stds + 1e-8).cpu().numpy().flatten(), batch_size).astype(np.float64)

        B = self.quantization_bits
        model_family = constriction.stream.model.QuantizedGaussian(-2**(B-1), 2**(B-1)-1)
        decoder = constriction.stream.stack.AnsCoder(x)
        symbols = decoder.decode(model_family, means, stds)
        return torch.tensor(symbols, dtype=torch.int32).reshape(original_shape).to(next(self.parameters()).device)

    def pca_rotation(self, x):
        # TODO proper PCA
        y = (x - self.z_means) / (self.z_stds+ 1e-8)
        return y

    def pca_inverse(self, x):
        # TODO proper PCA
        return x * self.z_stds + self.z_means

    def quantizer(self, x):
        B = self.quantization_bits
        x_quantized = torch.round(2**(B-1) * x).clamp(-2**(B-1), 2**(B-1)-1).to(torch.int32)
        return x_quantized

    def dequantizer(self, x):
        B = self.quantization_bits
        x_reconstructed = (x / 2**(B-1))
        return x_reconstructed

    def compute_priors(self, dataloader):
        print("Computing priors...")
        all_latents = []
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                z = self.pass_to_encoders(batch)
                all_latents.append(z)

        all_latents = torch.cat(all_latents, dim=0)
        z_means = all_latents.mean(dim=0)
        z_stds = all_latents.std(dim=0)
        self.register_buffer('z_means', z_means)
        self.register_buffer('z_stds', z_stds)

    def training_step(self, batch, batch_idx):
        x = batch
        z = self.pass_to_encoders(x)
        #z = self.encoder(x)

        # add uniform noise to simulate quantization
        noise = torch.zeros_like(z).uniform_(-(1.0/1024.0), 1.0/1024.0)

        # x_hat = self.decoder(z+noise)
        x_hat = self.pass_to_decoders(z+noise)

        loss = F.mse_loss(x_hat, x) + self.rate_coeffitient*torch.mean(z ** 2)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        # z = self.encoder(x)
        z = self.pass_to_encoders(x)
        # x_hat = self.decoder(z)
        x_hat = self.pass_to_decoders(z)

        loss = F.mse_loss(x_hat, x) + self.rate_coeffitient*torch.mean(z ** 2)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def forward(self, x):
        x_hat, _= self.forward_get_latent(x)
        return x_hat

    def forward_get_latent(self, x):
        # z = self.encoder(x)
        z = self.pass_to_encoders(x)
        z_rot = self.pca_rotation(z)
        z_q = self.quantizer(z_rot)

        USE_FANCY_COMPRESSION = False
        if USE_FANCY_COMPRESSION:
            z_compressed = self.entropy_coder(z_q)
            original_shape = z_q.shape
            z_decompressed = self.entropy_decoder(z_compressed, original_shape)
            #
            z_compressed_data = z_compressed.tobytes()
            #
        else:
            symbols = z_q.cpu().numpy().astype(np.int32).flatten()
            codec = dahuffman.HuffmanCodec.from_data(symbols)
            z_compressed = codec.encode(symbols)
            z_decompressed = torch.tensor(codec.decode(z_compressed), dtype=torch.int32).reshape(z_q.shape).to(next(self.parameters()).device)
            z_compressed_data = z_compressed

        z_deq = self.dequantizer(z_decompressed)
        z_inv_rot = self.pca_inverse(z_deq)
        # x_hat = self.decoder(z_inv_rot)
        x_hat = self.pass_to_decoders(z_inv_rot)

        return (x_hat, z_compressed_data)


def train_model(datamodule, experiment_name, epochs, learning_rate):
    model = BasicAE(learning_rate=learning_rate)

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

    # load best model weights
    best_model = BasicAE.load_from_checkpoint(checkpoint_callback.best_model_path)

    # compute priors from latents
    best_model.compute_priors(datamodule.train_dataloader())

    print(
        f"Training complete. Best model saved to checkpoints/{os.path.basename(checkpoint_callback.best_model_path)}"
    )

    print(f"Finished experiment: {experiment_name}")
    print("=" * 30)

    return best_model
