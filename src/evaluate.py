import os
import datetime

import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure,
)
from torchvision.utils import save_image
import matplotlib

matplotlib.use("Agg")

from data import DF2KDataModule, ClassImagesDataModule, MinecraftDataModule
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
        self.msssim_metric = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=1.0
        ).to(self.device)
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
        self.msssim_metric.update(reconstruction, original)
        self.num_batches += 1

    def finilize(self):
        self.avg_mse = self.total_mse / self.num_batches if self.num_batches else 0.0
        psnr_val = self.psnr_metric.compute()
        ssim_val = self.msssim_metric.compute()

        self.avg_psnr = (
            psnr_val.item() if torch.is_tensor(psnr_val) else float(psnr_val)
        )
        self.avg_msssim = (
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
        print(f"MS-SSIM: {self.avg_msssim:.4f}", file=file)
        print("=" * 30 + "\n", file=file)


def run_evaluation(model, datamodule, evaluation_name, n_images=30, n_save=5):
    print(f"\n--- Running evaluation for {evaluation_name} ---")
    datamodule.setup()
    test_dataloader = datamodule.test_dataloader()

    device = next(model.parameters()).device
    # model.float()
    model.eval()

    metrics_ours = ImageComparisonMetrics("original", "ours", device=device)
    metrics_cae = ImageComparisonMetrics("original", "cae_only", device=device)
    # JPEG qualities to evaluate
    jpeg_qualities = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    jpeg_metrics = {
        q: ImageComparisonMetrics(f"original", f"jpeg_q{q}", device=device)
        for q in jpeg_qualities
    }
    jpeg_bpps = {q: [] for q in jpeg_qualities}
    model_bpps = []

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
            for i, batch in enumerate(test_dataloader):
                if i >= n_images:
                    break

                # Assuming batch size 1 for full images, or first image of batch
                original_tensor = batch[0].to(device)  # (C, H, W)
                img_pil = TF.to_pil_image(original_tensor.cpu())
                num_pixels = original_tensor.shape[1] * original_tensor.shape[2]

                # 1. Model prediction
                result = model.evaluate_image(original_tensor)
                recon_ours = result["reconstruction"]
                recon_cae = result.get("cae_reconstruction", None)
                payload = result["compressed_payload"]

                # 2. Update model metrics
                metrics_ours.update(recon_ours, original_tensor)
                if recon_cae is not None:
                    metrics_cae.update(recon_cae, original_tensor)

                # 3. JPEG baselines at different qualities
                for q in jpeg_qualities:
                    jpeg_pil, jpeg_size = get_jpeg_image(img_pil, quality=q)
                    recon_jpeg = TF.to_tensor(jpeg_pil)
                    jpeg_metrics[q].update(recon_jpeg, original_tensor)
                    jpeg_bpps[q].append((jpeg_size * 8.0) / num_pixels)

                # 4. Calculate BPP for our model
                bpp = (len(payload) * 8.0) / num_pixels
                model_bpps.append(bpp)

                # Log per-image results to file
                print(
                    f"Image {i}: {original_tensor.shape[2]}x{original_tensor.shape[1]} | {bpp:.3f} bpp",
                    file=f, flush=True
                )

                if i < n_save:
                    # Print status to console
                    print(f"Saving comparison for image {i}...")
                    save_path = f"{experiment_dir}/{evaluation_name}_{i}_comparison.png"
                    # Create a comparison strip: Original | Ours
                    to_stack = [original_tensor.cpu(), recon_ours.cpu()]
                    # Optionally include CAE if it exists
                    if recon_cae is not None:
                        to_stack.insert(1, recon_cae.cpu())

                    comparison = torch.cat(to_stack, dim=2)
                    save_image(comparison, save_path)



        # Final summaries to file
        if metrics_cae.num_batches > 0:
            metrics_cae.print_summary(file=f, flush=True)

        metrics_ours.print_summary(file=f)

     

        # Calculate final averages
        metrics_ours.finilize()
        avg_bpp_ours = sum(model_bpps) / len(model_bpps)


        print("\n[RD_DATA]", file=f)
        print(f"model_bpp: {avg_bpp_ours:.6f}", file=f)
        print(f"model_psnr: {metrics_ours.avg_psnr:.4f}", file=f)
        print(f"model_ms-ssim: {metrics_ours.avg_msssim:.4f}", file=f)

        print("\n[JPEG_RD_CURVE]", file=f)
        for q in jpeg_qualities:
            jpeg_metrics[q].finilize()
            avg_bpp_q = sum(jpeg_bpps[q]) / len(jpeg_bpps[q])
            print(
                f"q={q}: bpp={avg_bpp_q:.6f}, psnr={jpeg_metrics[q].avg_psnr:.4f}, ms-ssim={jpeg_metrics[q].avg_msssim:.4f}",
                file=f,
            )


    print(f"Evaluation complete. Results saved to {experiment_dir}")


def main():
    # We use batch_size=1 and no crop for high-level evaluation on full images

    datamodule_full = DF2KDataModule(
        train_dir="datasets/DF2K/train",
        test_dir="datasets/DF2K/test",
        batch_size=1,
        ycbcr=False,
        random_crop=False,
        val_batch_size=1,
    )


    models = ["DCAL_LAB_flops_best.pt"]

    for model_name in models:
        try:
            model = torch.load(f"checkpoints/manual/{model_name}", weights_only=False)
            run_evaluation(model, datamodule_full, f"{model_name}_eval", n_images=30)
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")


if __name__ == "__main__":
    main()
