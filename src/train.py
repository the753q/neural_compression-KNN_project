import os
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from data import ClassImagesDataModule, DF2KDataModule
from models import get_model


torch.set_float32_matmul_precision("medium")

datamodule_default_imagenet10k = ClassImagesDataModule(
    data_dir="datasets/imagenet_10K/imagenet_subtrain",
    batch_size=64,
    num_workers=10,
    random_crop=True,
    ycbcr=True,
    patch_size=128,
)

datamodule_df2k = DF2KDataModule(
    train_dir="datasets/DF2K/train",
    test_dir="datasets/DF2K/test",
    batch_size=8,
    ycbcr=True,
    random_crop=True,
)


def experiment1():
    """
    Train a basic AE on ImageNet.
    """
    EXPERIMENT_NAME = "basic_imagenet10k"
    MODEL_NAME = "basic"
    EPOCHS = 15
    LEARNING_RATE = 2e-4

    model = get_model(MODEL_NAME, learning_rate=LEARNING_RATE)

    checkpoint_filename = f"{EXPERIMENT_NAME}-{MODEL_NAME}-best"

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=checkpoint_filename,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        precision="bf16-mixed",
        callbacks=[checkpoint_callback],
    )

    print("=" * 30)
    print(f"Started experiment: {EXPERIMENT_NAME}")

    print(f"Starting training for {MODEL_NAME}...")
    trainer.fit(model, datamodule_default_imagenet10k)
    print(
        f"Training complete. Best model saved to checkpoints/{os.path.basename(checkpoint_filename)}.ckpt"
    )

    print(f"Finished experiment: {EXPERIMENT_NAME}")
    print("=" * 30)

    # compute latents for quantization
    print("Computing priors...")
    all_latents = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in datamodule_default_imagenet10k.train_dataloader():
            batch = batch.to(device)
            z = model.encoder(batch)
            all_latents.append(z)

    all_latents = torch.cat(all_latents, dim=0)

    # load best model weights
    best_model = (model.__class__).load_from_checkpoint(
        checkpoint_callback.best_model_path
    )
    best_model.to(device)
    # compute priors from latents
    best_model.compute_priors(all_latents)
    # save model as torch object
    torch.save(best_model, f"checkpoints/manual/{MODEL_NAME}_best.pt")


def experiment2():
    """
    Train a basic DCAL 2018 on ImageNet..
    """
    EXPERIMENT_NAME = "dcal_df2k"
    MODEL_NAME = "DCAL_2018"
    EPOCHS = 20
    LEARNING_RATE = 1e-4

    model = get_model(MODEL_NAME, learning_rate=LEARNING_RATE)

    checkpoint_filename = f"{EXPERIMENT_NAME}-{MODEL_NAME}-best"

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=checkpoint_filename,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    csv_logger = CSVLogger("logs/", name=EXPERIMENT_NAME)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        callbacks=[checkpoint_callback],
        logger=csv_logger,
    )

    print("=" * 30)
    print(f"Started experiment: {EXPERIMENT_NAME}")

    print(f"Starting training for {MODEL_NAME}...")
    trainer.fit(model, datamodule_default_imagenet10k)
    print(
        f"Training complete. Best model saved to checkpoints/{os.path.basename(checkpoint_filename)}"
    )

    print(f"Finished experiment: {EXPERIMENT_NAME}")
    print("=" * 30)

    # load best model weights
    best_model = (model.__class__).load_from_checkpoint(
        checkpoint_callback.best_model_path
    )
    # save model as torch object
    torch.save(best_model, f"checkpoints/manual/{MODEL_NAME}_best.pt")

def experiment3():
    """
    Train balle_2016 on ImageNet..
    """
    EXPERIMENT_NAME = "balle_2016_imagenet"
    MODEL_NAME = "balle_2016"
    EPOCHS = 10
    LEARNING_RATE = 1e-4

    model = get_model(MODEL_NAME, learning_rate=LEARNING_RATE)

    checkpoint_filename = f"{EXPERIMENT_NAME}-{MODEL_NAME}-best"

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=checkpoint_filename,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    csv_logger = CSVLogger("logs/", name=EXPERIMENT_NAME)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        callbacks=[checkpoint_callback],
        logger=csv_logger,
    )

    print("=" * 30)
    print(f"Started experiment: {EXPERIMENT_NAME}")

    print(f"Starting training for {MODEL_NAME}...")
    trainer.fit(model, datamodule_default_imagenet10k)
    print(
        f"Training complete. Best model saved to checkpoints/{os.path.basename(checkpoint_filename)}"
    )

    print(f"Finished experiment: {EXPERIMENT_NAME}")
    print("=" * 30)

    # load best model weights
    best_model = (model.__class__).load_from_checkpoint(
        checkpoint_callback.best_model_path
    )
    # save model as torch object
    torch.save(best_model, f"checkpoints/manual/{MODEL_NAME}_best.pt")


def main():
    # pass
    # experiment1()
    experiment3()


if __name__ == "__main__":
    main()
