# Deep Convolutional AutoEncoder-based Lossy Image Compression - Native Inference
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import constriction
import numpy as np
import dahuffman
from utils import rgb_to_ycbcr, ycbcr_to_rgb
from training_utils import universal_train_model


class DownBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        N = [32, 32, 64, 64, 64, 64, 64, 32]

        def downsampling_unit(in_c, out_c1, out_c2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c1, kernel_size=3, stride=1, padding=1),
                nn.PReLU(out_c1),
                nn.Conv2d(out_c1, out_c2, kernel_size=3, stride=2, padding=1),
                nn.PReLU(out_c2),
            )

        self.net = nn.Sequential(
            downsampling_unit(in_channels, N[0], N[1]),
            downsampling_unit(N[1], N[2], N[3]),
            downsampling_unit(N[3], N[4], N[5]),
            downsampling_unit(N[5], N[6], N[7]),
        )

    def forward(self, x):
        return self.net(x)


class UpBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        N = [32, 32, 64, 64, 64, 64, 64, 32][::-1]

        def upsampling_unit(in_c, out_c1, out_c2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c1, kernel_size=3, stride=1, padding=1),
                nn.PReLU(out_c1),
                nn.ConvTranspose2d(out_c1, out_c2, kernel_size=4, stride=2, padding=1),
                nn.PReLU(out_c2),
            )

        self.net = nn.Sequential(
            upsampling_unit(in_channels, N[0], N[1]),
            upsampling_unit(N[1], N[2], N[3]),
            upsampling_unit(N[3], N[4], N[5]),
            upsampling_unit(N[5], N[6], N[7]),
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.row1 = DownBranch(1)
        self.row2 = DownBranch(1)
        self.row3 = DownBranch(1)

    def forward(self, x):
        y, cb, cr = torch.split(x, 1, dim=1)

        out1 = self.row1(y)
        out2 = self.row2(cb)
        out3 = self.row3(cr)

        return out1, out2, out3


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.row1 = UpBranch(32)
        self.row2 = UpBranch(32)
        self.row3 = UpBranch(32)

        self.final_merge = nn.Sequential(
            nn.Conv2d(96, 3, kernel_size=3, padding=1), nn.Sigmoid()
        )

    def forward(self, x_y, x_cb, x_cr):
        out1 = self.row1(x_y)
        out2 = self.row2(x_cb)
        out3 = self.row3(x_cr)

        x_concat = torch.cat([out1, out2, out3], dim=1)
        return self.final_merge(x_concat)


class DCAL_extended(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.name = "DCAL_extended"
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quantization_bits = 8  # lower -> more compression
        self.rate_coeffitient = 1.0  # higher -> more compression
        assert self.rate_coeffitient >= 0.0

    def entropy_coder(self, x):
        symbols = x.cpu().numpy().astype(np.int32).flatten()
        # tile to match batch dimension in symbols
        means = np.tile(self.z_means.cpu().numpy().flatten(), x.shape[0]).astype(
            np.float64
        )
        stds = np.tile((self.z_stds + 1e-8).cpu().numpy().flatten(), x.shape[0]).astype(
            np.float64
        )

        B = self.quantization_bits
        model_family = constriction.stream.model.QuantizedGaussian(
            -(2 ** (B - 1)), 2 ** (B - 1) - 1
        )
        coder = constriction.stream.stack.AnsCoder()
        coder.encode_reverse(symbols, model_family, means, stds)
        compressed = coder.get_compressed()

        return compressed

    def entropy_decoder(self, x, original_shape):
        # tile to match batch size
        batch_size = original_shape[0]
        means = np.tile(self.z_means.cpu().numpy().flatten(), batch_size).astype(
            np.float64
        )
        stds = np.tile((self.z_stds + 1e-8).cpu().numpy().flatten(), batch_size).astype(
            np.float64
        )

        B = self.quantization_bits
        model_family = constriction.stream.model.QuantizedGaussian(
            -(2 ** (B - 1)), 2 ** (B - 1) - 1
        )
        decoder = constriction.stream.stack.AnsCoder(x)
        symbols = decoder.decode(model_family, means, stds)
        return (
            torch.tensor(symbols, dtype=torch.int32)
            .reshape(original_shape)
            .to(next(self.parameters()).device)
        )

    def pca_rotation(self, y):
        B, N6, H, W = y.shape
        m = H * W

        # reshape to (B, N6, m) — each spatial location is one sample
        y_flat = y.view(B, N6, m)

        # covariance matrix (B, N6, N6)
        cov = (y_flat @ y_flat.transpose(1, 2)) / m

        # eigenvectors — eigenvalues sorted ascending, so flip
        eigenvalues, U = torch.linalg.eigh(cov)
        U = U.flip(-1)  # (B, N6, N6), columns sorted descending by eigenvalue

        # rotate
        y_rot = U.transpose(1, 2) @ y_flat  # (B, N6, m)
        y_rot = y_rot.view(B, N6, H, W)

        return y_rot, U

    def pca_inverse(self, y_rot, U):
        B, N6, H, W = y_rot.shape
        y_flat = y_rot.view(B, N6, H * W)
        y = U @ y_flat  # (B, N6, m)
        return y.view(B, N6, H, W)

    def quantizer(self, x):
        B = self.quantization_bits
        x_quantized = (
            torch.round(2 ** (B - 1) * x)
            .clamp(-(2 ** (B - 1)), 2 ** (B - 1) - 1)
            .to(torch.int32)
        )
        return x_quantized

    def dequantizer(self, x):
        B = self.quantization_bits
        x_reconstructed = x / 2 ** (B - 1)
        return x_reconstructed

    def training_step(self, batch, batch_idx):
        x = batch
        z_y, z_cb, z_cr = self.encoder(x)

        # add uniform noise to simulate quantization
        z = torch.cat([z_y, z_cb, z_cr], dim=1)
        noise = torch.zeros_like(z).uniform_(-(1.0 / 1024.0), 1.0 / 1024.0)
        z_y, z_cb, z_cr = torch.split(z + noise, 32, dim=1)

        x_hat = self.decoder(z_y, z_cb, z_cr)

        loss = F.mse_loss(x_hat, x) + self.rate_coeffitient * torch.mean(z**2)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        z_y, z_cb, z_cr = self.encoder(x)
        z = torch.cat([z_y, z_cb, z_cr], dim=1)

        x_hat = self.decoder(z_y, z_cb, z_cr)

        loss = F.mse_loss(x_hat, x) + self.rate_coeffitient * torch.mean(z**2)

        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def forward(self, x):
        x_hat, _ = self.forward_get_latent(x)
        return x_hat

    def forward_just_cae(self, x):
        z_Y, z_Cb, z_Cr = self.encoder(x)
        x_hat = self.decoder(z_Y, z_Cb, z_Cr)
        return x_hat

    def forward_get_latent(self, x):
        z_Y, z_Cb, z_Cr = self.encoder(x)

        # PCA rotation
        z_rot_Y, U_Y = self.pca_rotation(z_Y)
        z_rot_Cb, U_Cb = self.pca_rotation(z_Cb)
        z_rot_Cr, U_Cr = self.pca_rotation(z_Cr)

        # Truncation
        z_rot_Y[:, -12:] = 0
        z_rot_Cb[:, -26:] = 0
        z_rot_Cr[:, -26:] = 0

        # Quantization
        z_quant_Y = self.quantizer(z_rot_Y)
        z_quant_Cb = self.quantizer(z_rot_Cb)
        z_quant_Cr = self.quantizer(z_rot_Cr)

        U_quant_Y = self.quantizer(U_Y)
        U_quant_Cb = self.quantizer(U_Cb)
        U_quant_Cr = self.quantizer(U_Cr)

        # Entropy coding
        U_quant_join = torch.cat([U_quant_Y, U_quant_Cb, U_quant_Cr], dim=1)
        z_quant_join = torch.cat([z_quant_Y, z_quant_Cb, z_quant_Cr], dim=1)

        payload = np.concatenate(
            [
                z_quant_join.cpu().numpy().astype(np.int32).flatten(),
                U_quant_join.cpu().numpy().astype(np.int32).flatten(),
            ]
        )

        USE_FANCY_COMPRESSION = False
        if USE_FANCY_COMPRESSION:
            z_compressed = self.entropy_coder(z_quant_join)
            compressed_payload = z_compressed.tobytes()
        else:
            codec = dahuffman.HuffmanCodec.from_data(payload)
            compressed_payload = codec.encode(payload)

        # De-quantization
        U_rec_Y = self.dequantizer(U_quant_Y)
        U_rec_Cb = self.dequantizer(U_quant_Cb)
        U_rec_Cr = self.dequantizer(U_quant_Cr)

        z_rec_Y = self.dequantizer(z_quant_Y)
        z_rec_Cb = self.dequantizer(z_quant_Cb)
        z_rec_Cr = self.dequantizer(z_quant_Cr)
        # PCA inverse rotation
        z_inv_pca_Y = self.pca_inverse(z_rec_Y, U_rec_Y)
        z_inv_pca_Cb = self.pca_inverse(z_rec_Cb, U_rec_Cb)
        z_inv_pca_Cr = self.pca_inverse(z_rec_Cr, U_rec_Cr)

        x_hat = self.decoder(z_inv_pca_Y, z_inv_pca_Cb, z_inv_pca_Cr)

        return (x_hat, compressed_payload)

    def evaluate_image(self, x):
        """
        Native evaluation method supporting arbitrary image sizes.
        x: (C, H, W) tensor (RGB)
        Returns: dict with reconstructions and compressed data.
        """
        device = next(self.parameters()).device
        x = x.to(device)

        # Convert to YCbCr for processing
        x_ycbcr = rgb_to_ycbcr(x)
        
        # Get original dimensions
        c, h, w = x_ycbcr.shape
        
        # Calculate padding to make dimensions multiples of 8 (due to 3 downsampling stages)
        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8
        
        # Apply reflection padding (right and bottom)
        # F.pad expects (left, right, top, bottom) for 4D input
        x_padded = F.pad(x_ycbcr.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect')
        
        with torch.no_grad():
            # Process entire image natively
            recon_padded, compressed = self.forward_get_latent(x_padded)
            recon_cae_padded = self.forward_just_cae(x_padded)
            
        # Crop back to original size and remove batch dimension
        recon = recon_padded[:, :, :h, :w].squeeze(0)
        recon_cae = recon_cae_padded[:, :, :h, :w].squeeze(0)
        
        # Convert back to RGB for final output
        full_reconstruction = ycbcr_to_rgb(recon)
        full_reconstruction_cae = ycbcr_to_rgb(recon_cae)
        
        # Ensure compressed payload is in bytes for consistent size calculation
        if isinstance(compressed, np.ndarray):
            compressed_payload = compressed.tobytes()
        elif isinstance(compressed, list):
            compressed_payload = bytes(compressed)
        else:
            compressed_payload = compressed

        return {
            "reconstruction": full_reconstruction,
            "cae_reconstruction": full_reconstruction_cae,
            "compressed_payload": compressed_payload,
        }


def train_model(datamodule, experiment_name, epochs, learning_rate, target_flops=None):
    return universal_train_model(
        DCAL_extended,
        datamodule,
        experiment_name,
        epochs,
        learning_rate,
        target_flops=target_flops,
    )
