import os
import torch
from data import ClassImagesDataModule, DF2KDataModule, MinecraftDataModule
from models import get_train_function


torch.set_float32_matmul_precision("medium")

datamodule_default_imagenet10k = ClassImagesDataModule(
    data_dir="datasets/imagenet_10K/imagenet_subtrain",
    batch_size=64,
    num_workers=10,
    random_crop=True,
    ycbcr=True,
    patch_size=128,
)

datamodule_default_imagenet10k_LAB = ClassImagesDataModule(
    data_dir="datasets/imagenet_10K/imagenet_subtrain",
    batch_size=64,
    num_workers=10,
    random_crop=True,
    ycbcr=False,
    lab=True,
    patch_size=128,
)

datamodule_df2k = DF2KDataModule(
    train_dir="datasets/DF2K/train",
    test_dir="datasets/DF2K/test",
    batch_size=8,
    ycbcr=False,
    random_crop=True,
    patch_size=256,
    val_patch_size=640,
    val_batch_size=5,
)

datamodule_minecraft_screenshots = MinecraftDataModule(
    train_dir="datasets/minecraft_screenshots/train",
    test_dir="datasets/minecraft_screenshots/test",
    batch_size=8,
    ycbcr=False,
    random_crop=True,
    patch_size=256,
    val_patch_size=640,
    val_batch_size=5,
)


def experiment1():
    """
    Train a basic AE on ImageNet.
    """
    EXPERIMENT_NAME = "basic_imagenet10k"
    MODEL_NAME = "basic"
    EPOCHS = 15
    LEARNING_RATE = 2e-4

    train_fn = get_train_function(MODEL_NAME)
    best_model = train_fn(
        datamodule_default_imagenet10k, EXPERIMENT_NAME, EPOCHS, LEARNING_RATE
    )

    # save model as torch object
    os.makedirs("checkpoints/manual", exist_ok=True)
    torch.save(best_model, f"checkpoints/manual/{MODEL_NAME}_best.pt")


def experiment2():
    """
    Train a basic DCAL 2018 on ImageNet..
    """
    EXPERIMENT_NAME = "dcal_df2k"
    MODEL_NAME = "DCAL_2018"
    EPOCHS = 2
    LEARNING_RATE = 1e-4

    train_fn = get_train_function(MODEL_NAME)
    best_model = train_fn(
        datamodule_default_imagenet10k, EXPERIMENT_NAME, EPOCHS, LEARNING_RATE
    )

    # save model as torch object
    os.makedirs("checkpoints/manual", exist_ok=True)
    torch.save(best_model, f"checkpoints/manual/{MODEL_NAME}_best.pt")


def experiment3():
    """
    Train Balle 2017 model on ImageNet.
    """
    EXPERIMENT_NAME = "balle_imagenet10k"
    MODEL_NAME = "Balle2017"
    EPOCHS = 15
    LEARNING_RATE = 1e-4

    train_fn = get_train_function(MODEL_NAME)
    best_model = train_fn(
        datamodule_default_imagenet10k, EXPERIMENT_NAME, EPOCHS, LEARNING_RATE
    )

    # save model as torch object
    os.makedirs("checkpoints/manual", exist_ok=True)
    torch.save(best_model, f"checkpoints/manual/{MODEL_NAME}_best.pt")


def experiment4():
    """
    Train CustomCompressor model on ImageNet.
    """
    EXPERIMENT_NAME = "custom_imagenet10k"
    MODEL_NAME = "CustomCompressor"
    EPOCHS = 15
    LEARNING_RATE = 1e-4

    train_fn = get_train_function(MODEL_NAME)
    best_model = train_fn(
        datamodule_default_imagenet10k, EXPERIMENT_NAME, EPOCHS, LEARNING_RATE
    )

    # save model as torch object
    os.makedirs("checkpoints/manual", exist_ok=True)
    torch.save(best_model, f"checkpoints/manual/{MODEL_NAME}_best.pt")


def experiment55():
    """
    Train CustomCompressor model with a fixed computational budget (FLOPs).
    """
    EXPERIMENT_NAME = "DCAL_Native"
    MODEL_NAME = "DCAL_Native"
    EPOCHS = 100  # Will be overridden by flops limit
    LEARNING_RATE = 1e-4
    TARGET_FLOPS = 3e15

    train_fn = get_train_function(MODEL_NAME)
    best_model = train_fn(
        datamodule_default_imagenet10k,
        EXPERIMENT_NAME,
        EPOCHS,
        LEARNING_RATE,
        target_flops=TARGET_FLOPS,
    )

    # save model as torch object
    os.makedirs("checkpoints/manual", exist_ok=True)
    torch.save(best_model, f"checkpoints/manual/{MODEL_NAME}_flops_best.pt")


def experiment_lab():
    EXPERIMENT_NAME = "dcal_lab_imagenet"
    MODEL_NAME = "DCAL_LAB"
    EPOCHS = 100  # Will be overridden by flops limit
    LEARNING_RATE = 1e-4
    TARGET_FLOPS = 3e15

    train_fn = get_train_function(MODEL_NAME)
    best_model = train_fn(
        datamodule_default_imagenet10k_LAB,
        EXPERIMENT_NAME,
        EPOCHS,
        LEARNING_RATE,
        target_flops=TARGET_FLOPS,
    )

    # save model as torch object
    os.makedirs("checkpoints/manual", exist_ok=True)
    torch.save(best_model, f"checkpoints/manual/{MODEL_NAME}_flops_best.pt")


def experiment5():
    """
    Train a basic DCAL 2018 on DF2K..
    """
    EXPERIMENT_NAME = "dcal_df2k"
    MODEL_NAME = "DCAL_2018"
    EPOCHS = 100
    LEARNING_RATE = 1e-4

    train_fn = get_train_function(MODEL_NAME)
    best_model = train_fn(datamodule_df2k, EXPERIMENT_NAME, EPOCHS, LEARNING_RATE)

    # save model as torch object
    os.makedirs("checkpoints/manual", exist_ok=True)
    torch.save(best_model, f"checkpoints/manual/{MODEL_NAME}_best.pt")


def general_experiment(data):
    assert "experiment_name" in data
    assert "model_name" in data
    assert "epochs" in data
    assert "lr" in data
    assert "data_module" in data

    train_fn = get_train_function(data["model_name"])
    best_model = train_fn(
        data["data_module"], data["experiment_name"], data["epochs"], data["lr"]
    )

    # save model as torch object
    os.makedirs("checkpoints/manual", exist_ok=True)
    torch.save(best_model, f"checkpoints/manual/{data['experiment_name']}_best.pt")


def experiment_hyperprior(experiment_name, data_module, epochs, lr, lambda_):
    MODEL_NAME = "Hyperprior"

    train_fn = get_train_function(MODEL_NAME)
    best_model = train_fn(data_module, experiment_name, epochs, lr, lambda_)

    # save model as torch object
    os.makedirs("checkpoints/manual", exist_ok=True)
    torch.save(best_model, f"checkpoints/manual/{experiment_name}_best.pt")


def main():
    # general_experiment({
    #     "experiment_name": "hyperprior_df2k",
    #     "model_name": "Hyperprior",
    #     "epochs": 20,
    #     "lr": 1e-4,
    #     "data_module": datamodule_minecraft_screenshots
    # })
    experiment_hyperprior("hyperprior_df2k_001", datamodule_df2k, 100, 1e-4, 0.01)
    experiment_hyperprior("hyperprior_df2k_005", datamodule_df2k, 100, 1e-4, 0.05)
    experiment_hyperprior("hyperprior_df2k_01", datamodule_df2k, 100, 1e-4, 0.1)

    experiment_hyperprior(
        "hyperprior_minecraft_001", datamodule_minecraft_screenshots, 100, 1e-4, 0.01
    )
    experiment_hyperprior(
        "hyperprior_minecraft_005", datamodule_minecraft_screenshots, 100, 1e-4, 0.05
    )
    experiment_hyperprior(
        "hyperprior_minecraft_01", datamodule_minecraft_screenshots, 100, 1e-4, 0.1
    )


if __name__ == "__main__":
    main()
