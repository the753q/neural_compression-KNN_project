import os

import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import save_image

from PIL import Image, ImageOps
import io
from torchvision import transforms
import dahuffman


from data import ImageNetSubsetDataModule, ClassImagesDataModule, Div2KDataModule, ConcatDatasetsDataModule

from models import get_model

import argparse

OUTPUT_DIR = "outputs"

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

datamodule_no_crop_concat = ConcatDatasetsDataModule(
    [
        Div2KDataModule(train_dir="../datasets/DIV2K_train_HR",
                      val_dir="../datasets/DIV2K_train_HR", random_crop=False),
        ClassImagesDataModule(data_dir="../datasets/imagenet_subtrain", random_crop=False)
    ],
    batch_size=1
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
  
def eval_compression(model_name, model_checkpoint, datamodule):
    assert(datamodule.batch_size == 1) # prevent too big sizes in memmory

    datamodule.setup()
    val_loader = datamodule.val_dataloader()
    
    print("Running evaluation of compression...\n")

    print(f"Loading {model_name} from {model_checkpoint}...")
    model_class = get_model(model_name).__class__
    model = model_class.load_from_checkpoint(model_checkpoint)

    model.eval()
    model.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    img_comp_metrics = ImageComparisonMetrics()

    N_IMAGES = 4
    print("="*45)
    print(f"{"i"}\t{'Image size'}\t{'Size before'}\t{'Size after'}\t{'ratio'}")
    print("="*45)
    for i, batch_tensor in enumerate(val_loader):
        if i+1 > N_IMAGES:
            break

        img = transforms.ToPILImage()(batch_tensor[0]).convert("RGB")

        patches = non_overlaping_patches(img)

        transform = transforms.ToTensor()
        patches_batch = torch.stack([transform(patch) for _, patch in patches]).to(device)

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

        print(f"{i}\t{img.size[0]}x{img.size[1]}\t{img_size//1024} KB \t{compressed_size//1024} KB \t{ratio:>15.2f}x")

        reconstructed = Image.new("RGB", img.size)
        for rec_patch, ((x,y), _) in zip(reconstructions, patches):
            patch = transforms.ToPILImage()(rec_patch.cpu())
            reconstructed.paste(patch, (x, y))

        img_comp_metrics.update(transform(img).unsqueeze(0), transform(reconstructed).unsqueeze(0))

        reconstructed.save(f"{OUTPUT_DIR}/{model_checkpoint.replace("/", "_")}_{i}_reconstructed.png")

    img_comp_metrics.print_summary()

def eval_patches(model_name, model_checkpoint, datamodule):
    datamodule.setup()
    val_loader = datamodule.val_dataloader()

    print("Running evaluation by patches... \n")

    print(f"Loading {model_name} from {model_checkpoint}...")
    model_class = get_model(model_name).__class__
    model = model_class.load_from_checkpoint(model_checkpoint)

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
    save_path = f"outputs/{model_checkpoint.replace("/", "_")}_comparison.png"
    save_image(comparison, save_path, nrow=first_batch_originals.shape[0])
    print(f"Image saved to {save_path}")
 
def main():
    os.makedirs("outputs", exist_ok=True)
    # eval_patches("basic", "checkpoints/basic_imagenet10k-basic-best.ckpt", datamodule_default_imagenet10k)
    eval_patches("DCAL_2018", "checkpoints/dcal_combined-DCAL_2018-best.ckpt", datamodule_default_concat)
    eval_compression("DCAL_2018", "checkpoints/dcal_combined-DCAL_2018-best.ckpt", datamodule_no_crop_concat)

if __name__ == "__main__":
    main()
