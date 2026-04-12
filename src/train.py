import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from data import ImageNetSubsetDataModule, ClassImagesDataModule, Div2KDataModule, ConcatDatasetsDataModule

from models import get_model

torch.set_float32_matmul_precision("medium")

datamodule_default_div2k = Div2KDataModule(
    train_dir="../datasets/DIV2K_train_HR",
    val_dir="../datasets/DIV2K_train_HR",
    batch_size=8
)

datamodule_default_imagenet10k = ClassImagesDataModule(
    data_dir="../datasets/imagenet_subtrain",
    batch_size=8,
    random_crop=True
)

datamodule_default_concat = ConcatDatasetsDataModule(
    [datamodule_default_div2k, datamodule_default_imagenet10k],
    batch_size=8
)


def experiment1():
    """
        Train a basic AE on ImageNet.
    """
    EXPERIMENT_NAME = "basic_imagenet10k"
    MODEL_NAME = "basic"
    EPOCHS = 5
    LEARNING_RATE = 1e-3
    
    model = get_model(MODEL_NAME, learning_rate=LEARNING_RATE)

    checkpoint_filename = f"{EXPERIMENT_NAME}-{MODEL_NAME}-best"

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=checkpoint_filename,
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        callbacks=[checkpoint_callback],
    )

    print("="*30)
    print(f"Started experiment: {EXPERIMENT_NAME}")

    print(f"Starting training for {MODEL_NAME}...")
    trainer.fit(model, datamodule_default_imagenet10k)
    print(f"Training complete. Best model saved to checkpoints/{checkpoint_filename}.ckpt")

    print(f"Finished experiment: {EXPERIMENT_NAME}")
    print("="*30)

def experiment2():
    """
        Train a basic DCAL 2018 on div2K and imagenet combined.
    """
    EXPERIMENT_NAME = "dcal_combined"
    MODEL_NAME = "DCAL_2018"
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    
    model = get_model(MODEL_NAME, learning_rate=LEARNING_RATE)

    checkpoint_filename = f"{EXPERIMENT_NAME}-{MODEL_NAME}-best"

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=checkpoint_filename,
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        callbacks=[checkpoint_callback],
    )

    print("="*30)
    print(f"Started experiment: {EXPERIMENT_NAME}")

    print(f"Starting training for {MODEL_NAME}...")
    trainer.fit(model, datamodule_default_concat)
    print(f"Training complete. Best model saved to checkpoints/{checkpoint_filename}.ckpt")

    print(f"Finished experiment: {EXPERIMENT_NAME}")
    print("="*30)

def main():
    experiment1()
    #experiment2()

if __name__ == "__main__":
    main()
