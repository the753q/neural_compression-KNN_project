import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from data import ImageNetSubsetDataModule, ImageNet10KDataModule
from models import get_model

MODEL_NAME = "basic"
DATA_DIR = "../datasets/imagenet_10k"
EPOCHS = 5
LEARNING_RATE = 1e-3

torch.set_float32_matmul_precision("medium")


def main():
    datamodule = ImageNet10KDataModule(
        data_dir=DATA_DIR,
        batch_size=8
    )

    model = get_model(MODEL_NAME, learning_rate=LEARNING_RATE)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=f"{MODEL_NAME}-best",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        callbacks=[checkpoint_callback],
    )

    print(f"Starting training for {MODEL_NAME}...")
    trainer.fit(model, datamodule)
    print(f"Training complete. Best model saved to checkpoints/{MODEL_NAME}-best.ckpt")


if __name__ == "__main__":
    main()
