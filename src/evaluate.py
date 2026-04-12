import os

import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import save_image

from PIL import Image, ImageOps
import io
from torchvision import transforms
import dahuffman

from data import ImageNetSubsetDataModule, ImageNet10KDataModule
from models import get_model

MODEL = "basic"
CKPT = "checkpoints/basic-best.ckpt"
DATA_DIR = "../datasets/imagenet_subtrain"
NUM_IMAGES = 8
OUTPUT_DIR = "outputs"

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

def non_overlaping_patches(img):
    """
        Creates non-overlapping patches of the image.
        Returns:
            list: list of tuples ((x,y), image_data)
    """
    patch_size = 256

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

def quantize_tensor(tensor):
    """
        Quantizes tensor using IQR.
        Returns:
            tuple: (quantized, reversed)
    """
    q25 = torch.quantile(tensor, 0.25)
    q75 = torch.quantile(tensor, 0.75)
    iqr = q75 - q25
    low  = q25 - 1.5 * iqr
    high = q75 + 1.5 * iqr

    tensor = tensor.clamp(low, high) # clamp
    tensor = (tensor - low) / (high - low) # normalize

    tensor_u8 = (tensor * 255).round().to(torch.uint8)

    tensor_rec = (tensor_u8.to(torch.float) / 255.0) * (high - low) + low

    return (tensor_u8, tensor_rec)
  
def eval_compression():
    print("Running evaluation of compression...\n")

    # FILE = "../datasets/misc/div2k_greece.png"
    FILE = "../datasets/misc/imagenet_birds.JPEG"

    img = Image.open(FILE).convert("RGB")
    patches =  non_overlaping_patches(img)

    print(f"Loading {MODEL} from {CKPT}...")
    model_class = get_model(MODEL).__class__
    model = model_class.load_from_checkpoint(CKPT)

    model.eval()
    model.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    transform = transforms.ToTensor()
    patches_batch = torch.stack([transform(patch) for _, patch in patches]).to(device)

    img_comp_metrics = ImageComparisonMetrics()

    quantized_tensor = None
    with torch.no_grad():
        bottleneck = model.encoder(patches_batch)

        quantized_tensor, tensor_reconstructed = quantize_tensor(bottleneck)

        reconstructions = model.decoder(tensor_reconstructed)

    # compare compression
    buf_img = io.BytesIO()
    img.save(buf_img, format="PNG")

    codec = dahuffman.HuffmanCodec.from_data(quantized_tensor.flatten().tolist())
    encoded = codec.encode(quantized_tensor.flatten().tolist())
    buf_compressed = io.BytesIO(encoded)

    img_size = buf_img.tell()
    compressed_size = buf_compressed.getbuffer().nbytes
    ratio = img_size / compressed_size

    print("-"*45)
    print(f"File: {FILE}, size: {img.size[0]}x{img.size[1]}")
    print(f"{'Size before':>15} | {'Size after':>15} | {'Compression ratio':>15}")
    print(f"{img_size//1024:>12} KB | {compressed_size//1024:>12} KB | {ratio:>15.2f}x")

    reconstructed = Image.new("RGB", img.size)
    for i, (x, y) in enumerate(pos for pos, _ in patches):
        patch = transforms.ToPILImage()(reconstructions[i].cpu())
        reconstructed.paste(patch, (x, y))

    img_comp_metrics.update(transform(img).unsqueeze(0), transform(reconstructed).unsqueeze(0))
    img_comp_metrics.print_summary()

    reconstructed.save(f"{OUTPUT_DIR}/reconstructed.png")

def eval_patches():
    os.makedirs("outputs", exist_ok=True)

    datamodule = ImageNet10KDataModule(
        data_dir=DATA_DIR,
        batch_size=NUM_IMAGES
    )
    datamodule.setup()
    val_loader = datamodule.val_dataloader()

    print("Running evaluation by patches... \n")

    print(f"Loading {MODEL} from {CKPT}...")
    model_class = get_model(MODEL).__class__
    model = model_class.load_from_checkpoint(CKPT)

    model.eval()
    model.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    img_comp_metrics = ImageComparisonMetrics()

    first_batch_originals = torch.empty(0)
    first_batch_reconstructions = torch.empty(0)

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
    save_path = f"outputs/{MODEL}_comparison.png"
    save_image(comparison, save_path, nrow=NUM_IMAGES)
    print(f"Image saved to {save_path}")
 
def main():
    eval_compression()
    eval_patches()

if __name__ == "__main__":
    main()
