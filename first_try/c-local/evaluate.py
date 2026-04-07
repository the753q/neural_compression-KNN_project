import argparse
import os

import torch
from torchvision.utils import save_image

from data import ImageNetSubsetDataModule
from models import get_model


def main(args):
    os.makedirs("outputs", exist_ok=True)

    datamodule = ImageNetSubsetDataModule(
        data_dir=args.data_dir,
        batch_size=args.num_images,
        image_size=args.image_size,
        val_limit=100
    )
    datamodule.setup()

    val_loader = datamodule.val_dataloader()
    originals = next(iter(val_loader))

    print(f"Loading {args.model} from {args.ckpt}...")

    model_class = get_model(args.model).__class__
    model = model_class.load_from_checkpoint(args.ckpt)

    model.eval()
    model.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    originals = originals.to(device)

    print("Generating reconstructions...")
    with torch.no_grad():
        reconstructions = model(originals)

    comparison = torch.cat([originals, reconstructions])

    save_path = f"outputs/{args.model}_comparison.png"

    save_image(comparison, save_path, nrow=args.num_images)
    print(f"Image saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="basic", help="Model name")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--data_dir", type=str, default="datasets/imagenet-mini")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_images", type=int, default=8, help="How many images to compare")
    args = parser.parse_args()
    
    main(args)
