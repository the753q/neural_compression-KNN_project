import os

import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import save_image

from PIL import Image, ImageOps
import io
from torchvision import transforms
import dahuffman

from data import DF2KDataModule, ClassImagesDataModule

from models import get_model

import argparse

OUTPUT_DIR = "outputs"

datamodule_imagenet10k_crop = ClassImagesDataModule(
    data_dir="datasets/imagenet_10K/imagenet_subtrain",
    batch_size=8,
    random_crop=True,
    ycbcr=True,
    patch_size=128
)

datamodule_imagenet10k_no_crop = ClassImagesDataModule(
    data_dir="datasets/imagenet_10K/imagenet_subtrain",
    batch_size=1,
    random_crop=False,
    ycbcr=True,
)

datamodule_df2k_crop = DF2KDataModule(
    train_dir="datasets/DF2K/train",
    test_dir="datasets/DF2K/test",
    batch_size=8,
    random_crop=True,
    ycbcr=True,
)

datamodule_df2k_no_crop = DF2KDataModule(
    train_dir="datasets/DF2K/train",
    test_dir="datasets/DF2K/test",
    batch_size=1,
    random_crop=False,
    ycbcr=True,
)

class ImageComparisonMetrics:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reset()

    def reset(self):
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.total_mse = 0.0
        self.num_batches = 0
        self.finilized = False

    def update(self, reconstruction, original):
        assert(not self.finilized)
        self.total_mse += F.mse_loss(reconstruction, original).item()
        self.psnr_metric.update(reconstruction, original)
        self.ssim_metric.update(reconstruction, original)
        self.num_batches += 1

    def finilize(self):
        self.avg_mse = self.total_mse / self.num_batches
        self.avg_psnr = self.psnr_metric.compute()
        self.avg_ssim = self.ssim_metric.compute()
        self.finilized = True

    def print_summary(self):
        if not self.finilized:
            self.finilize()
        print("\n" + "=" * 30)
        print(f"Total batches: {self.num_batches}")
        print(f"MSE:  {self.avg_mse:.6f}")
        print(f"PSNR: {self.avg_psnr:.2f} dB")
        print(f"SSIM: {self.avg_ssim:.4f}")
        print("=" * 30 + "\n")

class ImagePatcher:
    def __init__(self, patch_size):
        self.patch_size = patch_size
        pass
    
    def create_patches(self, img):
        """
        Creates non-overlapping patches of the image.
        Returns:
            list: list of tuples ((x,y), image_data)
        """
        patch_size = self.patch_size

        # pad to create non-overlaping patches
        pad_w = (patch_size - img.size[0] % patch_size) % patch_size
        pad_h = (patch_size - img.size[1] % patch_size) % patch_size

        img = ImageOps.expand(img, (0, 0, pad_w, pad_h))

        patches = []

        for y in range(0, img.size[1], patch_size):
            for x in range(0, img.size[0], patch_size):
                patch_img = img.crop((x, y, x + patch_size, y + patch_size))
                patch = ((x,y), patch_img)
                patches.append(patch)

        return patches

    def combine_patches(self, img_size, positions, patches):
        reconstructed = Image.new("RGB", img_size)
        for (x,y), patch in zip(positions, patches):
            reconstructed.paste(patch, (x, y))
        return reconstructed

def eval_compression(model, evaluation_name, datamodule):
    assert(datamodule.batch_size == 1) # prevent too big sizes in memmory

    datamodule.setup()
    val_loader = datamodule.val_dataloader()
    
    print("Running evaluation of compression...\n")

    device = next(model.parameters()).device

    img_comp_metrics_ours = ImageComparisonMetrics()
    img_comp_metrics_jpeg = ImageComparisonMetrics()
    img_patcher = ImagePatcher(patch_size=128)

    N_IMAGES = 4
    print("="*45)
    print(f"{"i"}\t{'Image size'}\t{'Size before'}\t{'Size after'}\t{'ratio'}\t{'jpeg_ratio'}")
    print("="*45)
    for i, batch_tensor in enumerate(val_loader):
        if i+1 > N_IMAGES:
            break

        img = transforms.ToPILImage()(batch_tensor[0]).convert("RGB")

        patches = img_patcher.create_patches(img)

        transform = transforms.ToTensor()
        patches_batch = torch.stack([transform(patch) for _, patch in patches]).to(device)

        with torch.no_grad():
            reconstructions, bottleneck = model.forward_get_latent(patches_batch)

        # compare compression
        buf_img = io.BytesIO()
        img.save(buf_img, format="PNG")

        buf_compressed = io.BytesIO(bottleneck)

        img_size = buf_img.tell()
        compressed_size = buf_compressed.getbuffer().nbytes
        ratio = img_size / compressed_size

        buf_jpeg = io.BytesIO()
        img.save(buf_jpeg, format="JPEG", quality=95)
        jpeg_size = buf_jpeg.tell()
        buf_jpeg.seek(0)
        jpeg_img = Image.open(buf_jpeg)

        jpeg_ratio = img_size/jpeg_size

        print(f"{i}\t{img.size[0]}x{img.size[1]}\t{img_size//1024} KB \t{compressed_size//1024} KB \t{ratio:>15.2f}x \t {jpeg_ratio:>15.2}x")

        reconstructed = img_patcher.combine_patches(img.size,
                        [(x,y) for (x,y), _ in patches],
                        [transforms.ToPILImage()(r.cpu()) for r in reconstructions])

        img_comp_metrics_ours.update(transform(img).unsqueeze(0), transform(reconstructed).unsqueeze(0))
        img_comp_metrics_jpeg.update(transform(img).unsqueeze(0), transform(jpeg_img).unsqueeze(0))

        reconstructed.save(f"{OUTPUT_DIR}/{evaluation_name}_{i}_reconstructed.png")

    print("\nOur comparison metrics:")
    img_comp_metrics_ours.print_summary()
    print("\nJPEG comparison metrics:")
    img_comp_metrics_jpeg.print_summary()

def eval_patches(model, evaluation_name, datamodule):
    datamodule.setup()
    val_loader = datamodule.val_dataloader()

    print("Running evaluation by patches... \n")

    img_comp_metrics = ImageComparisonMetrics()

    first_batch_originals = torch.empty(0)
    first_batch_reconstructions = torch.empty(0)

    device = next(model.parameters()).device

    print("Evaluating model over the validation set...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            originals = batch.to(device)
            reconstructions = model(originals)

            img_comp_metrics.update(reconstructions, originals)

            if i == 0:
                first_batch_originals = originals
                first_batch_reconstructions = reconstructions

    img_comp_metrics.print_summary()

    comparison = torch.cat([first_batch_originals, first_batch_reconstructions])
    save_path = f"outputs/{evaluation_name}_comparison.png"
    save_image(comparison, save_path, nrow=first_batch_originals.shape[0])
    print(f"Image saved to {save_path}")
 
def load_model_from_checkpoint(model_name, model_checkpoint):
    print(f"Loading {model_name} from {model_checkpoint}...")
    model_class = get_model(model_name).__class__
    model = model_class.load_from_checkpoint(model_checkpoint)

    model.eval()
    model.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model

def main():
    os.makedirs("outputs", exist_ok=True)

    #TODO get rid of need for crop datamodule in eval_patches

    # basic_model = load_model_from_checkpoint("basic", "checkpoints/basic_imagenet10k-basic-best-v1.ckpt")
    basic_model = torch.load("checkpoints/manual/basic_best.pt", weights_only=False)

    #eval_patches(basic_model, "basic_eval", datamodule_imagenet10k_crop)
    eval_compression(basic_model, "basic_eval", datamodule_imagenet10k_no_crop)
    #eval_compression("basic", "checkpoints/basic_imagenet10k-basic-best.ckpt", datamodule_imagenet10k_no_crop)
    

    #eval_patches("DCAL_2018", "checkpoints/dcal_combined-DCAL_2018-best.ckpt", datamodule_default_concat)
    #eval_compression("DCAL_2018", "checkpoints/dcal_combined-DCAL_2018-best.ckpt", datamodule_no_crop_concat)

if __name__ == "__main__":
    main()
