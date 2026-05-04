import lightning.pytorch as pl
import torch
import torch.nn.functional as F


class BaseAutoencoder(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.name = "unnamed_model"
        self.stride_requirement = 1

    def _pad(self, x):
        h, w = x.shape[-2:]
        s = self.stride_requirement
        pad_h = (s - h % s) % s
        pad_w = (s - w % s) % s
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, (h, w)

    def _unpad(self, x, original_size):
        h, w = original_size
        return x[..., :h, :w]

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement the forward pass")

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
