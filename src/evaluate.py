import os
import datetime

import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import save_image
import matplotlib

matplotlib.use("Agg")

from data import DF2KDataModule, ClassImagesDataModule
from models import get_model
from utils import get_jpeg_image

OUTPUT_DIR = "outputs"


class ImageComparisonMetrics:
    def __init__(self, name_1, name_2, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.name_1 = name_1
        self.name_2 = name_2
        self.reset()

    def reset(self):
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(
            self.device
        )
        self.total_mse = 0.0
        self.num_batches = 0
        self.finilized = False

    def update(self, reconstruction, original):
        assert not self.finilized
        # Ensure same device and shape (B, C, H, W)
        reconstruction = reconstruction.to(self.device)
        original = original.to(self.device)
        if reconstruction.dim() == 3:
            reconstruction = reconstruction.unsqueeze(0)
        if original.dim() == 3:
            original = original.unsqueeze(0)

        self.total_mse += F.mse_loss(reconstruction, original).item()
        self.psnr_metric.update(reconstruction, original)
        self.ssim_metric.update(reconstruction, original)
        self.num_batches += 1

    def finilize(self):
        self.avg_mse = self.total_mse / self.num_batches if self.num_batches else 0.0
        psnr_val = self.psnr_metric.compute()
        ssim_val = self.ssim_metric.compute()

        self.avg_psnr = (
            psnr_val.item() if torch.is_tensor(psnr_val) else float(psnr_val)
        )
        self.avg_ssim = (
            ssim_val.item() if torch.is_tensor(ssim_val) else float(ssim_val)
        )
        self.finilized = True

    def print_summary(self, file=None):
        if not self.finilized:
            self.finilize()
        print("\n" + "=" * 30, file=file)
        print(f"Image metrics comparison | {self.name_1} vs {self.name_2}", file=file)
        print("=" * 30, file=file)
        print(f"Total batches: {self.num_batches}", file=file)
        print(f"MSE:  {self.avg_mse:.6f}", file=file)
        print(f"PSNR: {self.avg_psnr:.2f} dB", file=file)
        print(f"SSIM: {self.avg_ssim:.4f}", file=file)
        print("=" * 30 + "\n", file=file)


def run_evaluation(model, datamodule, evaluation_name, n_images=30, n_save=5):
    print(f"\n--- Running evaluation for {evaluation_name} ---")
    datamodule.setup()
    val_loader = datamodule.val_dataloader()

    device = next(model.parameters()).device
    model.eval()

    metrics_ours = ImageComparisonMetrics("original", "ours", device=device)
    metrics_cae = ImageComparisonMetrics("original", "cae_only", device=device)
    metrics_jpeg = ImageComparisonMetrics("original", "jpeg", device=device)

    # Create timestamped output directory
    now = datetime.datetime.now().strftime("%m_%d_%H_%M")
    experiment_dir = os.path.join(OUTPUT_DIR, f"{evaluation_name}_{now}")
    os.makedirs(experiment_dir, exist_ok=True)
    results_path = os.path.join(experiment_dir, "results.txt")

    with open(results_path, "w") as f:
        print(f"Evaluation: {evaluation_name}", file=f)
        print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file=f)
        print("-" * 30, file=f)

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= n_images:
                    break

                # Assuming batch size 1 for full images, or first image of batch
                original_tensor = batch[0].to(device)  # (C, H, W)

                # 1. Model prediction
                result = model.evaluate_image(original_tensor)
                recon_ours = result["reconstruction"]
                recon_cae = result.get("cae_reconstruction", None)
                payload = result["compressed_payload"]

                # 2. JPEG baseline
                # Convert tensor to PIL for JPEG utility
                img_pil = TF.to_pil_image(original_tensor.cpu())
                jpeg_pil, jpeg_size = get_jpeg_image(img_pil)
                recon_jpeg = TF.to_tensor(jpeg_pil)

                # 3. Update metrics
                metrics_ours.update(recon_ours, original_tensor)
                if recon_cae is not None:
                    metrics_cae.update(recon_cae, original_tensor)
                metrics_jpeg.update(recon_jpeg, original_tensor)

                # 4. Calculate BPP and stats for this image
                bpp = (len(payload) * 8.0) / (
                    original_tensor.shape[1] * original_tensor.shape[2]
                )

                # Log per-image results to file
                print(
                    f"Image {i}: {original_tensor.shape[2]}x{original_tensor.shape[1]} | {bpp:.3f} bpp",
                    file=f,
                )

                if i < n_save:
                    # Print status to console
                    print(f"Saving comparison for image {i}...")
                    save_path = f"{experiment_dir}/{evaluation_name}_{i}_comparison.png"
                    # Create a comparison strip: Original | CAE | Ours
                    to_stack = [original_tensor.cpu(), recon_ours.cpu()]
                    if recon_cae is not None:
                        to_stack.insert(1, recon_cae.cpu())

                    comparison = torch.cat(to_stack, dim=2)
                    save_image(comparison, save_path)

        # Final summaries to file
        if metrics_cae.num_batches > 0:
            metrics_cae.print_summary(file=f)
        metrics_ours.print_summary(file=f)
        metrics_jpeg.print_summary(file=f)

    print(f"Evaluation complete. Results saved to {experiment_dir}")


def main():
    # We use batch_size=1 and no crop for high-level evaluation on full images
    datamodule_full = ClassImagesDataModule(
        data_dir="datasets/imagenet_10K/imagenet_subtrain",
        batch_size=1,
        random_crop=False,
        ycbcr=False,  # Standardized to RGB for eval loader
    )

    model_name = "DCAL_Native_flops_best.pt"
    model = torch.load(f"checkpoints/manual/{model_name}", weights_only=False)
    run_evaluation(model, datamodule_full, f"{model_name}_eval")


if __name__ == "__main__":
    main()
