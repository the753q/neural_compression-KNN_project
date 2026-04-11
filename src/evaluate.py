import os

import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import save_image

from data import ImageNetSubsetDataModule
from models import get_model

MODEL = "basic"
CKPT = "checkpoints/basic-best.ckpt"
DATA_DIR = "datasets/imagenet-mini"
IMAGE_SIZE = 256
NUM_IMAGES = 8


def main():
    os.makedirs("outputs", exist_ok=True)

    datamodule = ImageNetSubsetDataModule(
        data_dir=DATA_DIR,
        batch_size=NUM_IMAGES,
        image_size=IMAGE_SIZE,
        val_limit=100
    )
    datamodule.setup()
    val_loader = datamodule.val_dataloader()

    print(f"Loading {MODEL} from {CKPT}...")
    model_class = get_model(MODEL).__class__
    model = model_class.load_from_checkpoint(CKPT)

    model.eval()
    model.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    total_mse = 0.0
    num_batches = 0

    first_batch_originals = torch.empty(0)
    first_batch_reconstructions = torch.empty(0)

    print("Evaluating model over the validation set...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            originals = batch.to(device)
            reconstructions = model(originals)

            total_mse += F.mse_loss(reconstructions, originals).item()
            psnr_metric.update(reconstructions, originals)
            ssim_metric.update(reconstructions, originals)
            num_batches += 1

            if i == 0:
                first_batch_originals = originals
                first_batch_reconstructions = reconstructions

    avg_mse = total_mse / num_batches
    avg_psnr = psnr_metric.compute()
    avg_ssim = ssim_metric.compute()

    print("\n" + "=" * 30)
    print(f"MSE:  {avg_mse:.6f}")
    print(f"PSNR: {avg_psnr:.2f} dB")
    print(f"SSIM: {avg_ssim:.4f}")
    print("=" * 30 + "\n")

    comparison = torch.cat([first_batch_originals, first_batch_reconstructions])
    save_path = f"outputs/{MODEL}_comparison.png"
    save_image(comparison, save_path, nrow=NUM_IMAGES)
    print(f"Image saved to {save_path}")


if __name__ == "__main__":
    main()
