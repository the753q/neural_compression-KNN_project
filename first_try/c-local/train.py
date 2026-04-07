import argparse
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from data import ImageNetSubsetDataModule
from models import get_model

torch.set_float32_matmul_precision("medium")


def main(args):
    datamodule = ImageNetSubsetDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size
    )

    model = get_model(args.model, learning_rate=args.lr)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=f"{args.model}-best",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        callbacks=[checkpoint_callback],
    )

    print(f"Starting training for {args.model}...")
    trainer.fit(model, datamodule)
    print(f"Training complete. Best model saved to checkpoints/{args.model}-best.ckpt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="basic", help="Name of model from registry")
    parser.add_argument("--data_dir", type=str, default="datasets/imagenet-mini")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    main(args)