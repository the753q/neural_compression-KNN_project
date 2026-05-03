import os
import io
import matplotlib
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import save_image
from PIL import Image, ImageOps


matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import DF2KDataModule, ClassImagesDataModule
from models import get_model


OUTPUT_DIR = "outputs"


def get_jpeg_image(img):
    buf_jpeg = io.BytesIO()
    img.save(buf_jpeg, format="JPEG", quality=95)
    jpeg_size = buf_jpeg.tell()
    buf_jpeg.seek(0)
    jpeg_img = Image.open(buf_jpeg)
    return jpeg_img, jpeg_size


datamodule_imagenet10k_crop = ClassImagesDataModule(
    data_dir="datasets/imagenet_10K/imagenet_subtrain",
    batch_size=8,
    random_crop=True,
    ycbcr=True,
    patch_size=128,
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
    def __init__(self, name_1, name_2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.total_mse += F.mse_loss(reconstruction, original).item()
        self.psnr_metric.update(reconstruction, original)
        self.ssim_metric.update(reconstruction, original)
        self.num_batches += 1

    def finilize(self):
        self.avg_mse = self.total_mse / self.num_batches if self.num_batches else 0.0

        # torchmetrics may return tensors or tuples; coerce to Python floats
        psnr_val = self.psnr_metric.compute()
        ssim_val = self.ssim_metric.compute()

        if isinstance(psnr_val, tuple):
            psnr_val = psnr_val[0]
        if isinstance(ssim_val, tuple):
            ssim_val = ssim_val[0]

        self.avg_psnr = (
            psnr_val.item() if torch.is_tensor(psnr_val) else float(psnr_val)
        )
        self.avg_ssim = (
            ssim_val.item() if torch.is_tensor(ssim_val) else float(ssim_val)
        )
        self.finilized = True

    def print_summary(self):
        if not self.finilized:
            self.finilize()
        print("\n" + "=" * 30)
        print(f"Image mentrics comparison | {self.name_1} vs {self.name_2}")
        print("=" * 30)
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
                patch = ((x, y), patch_img)
                patches.append(patch)

        return patches

    def combine_patches(self, img_size, positions, patches):
        reconstructed = Image.new("RGB", img_size)
        for (x, y), patch in zip(positions, patches):
            reconstructed.paste(patch, (x, y))
        return reconstructed


class ImageCompressionMetric:
    _psnr_fn = PeakSignalNoiseRatio(data_range=1.0)

    def __init__(self, img_name, img_before, compressed, img_after):
        self.img_name = img_name
        self.dims = img_before.size

        buf_img = io.BytesIO()
        img_before.save(buf_img, format="PNG")
        buf_compressed = io.BytesIO(compressed)
        self.size_before = buf_img.tell()
        self.size_after = buf_compressed.getbuffer().nbytes
        self.ratio = self.size_before / self.size_after
        self.bpp = (self.size_after * 8.0) / (self.dims[0] * self.dims[1])

        jpeg_img, jpeg_size = get_jpeg_image(img_before)
        self.jpeg_ratio = self.size_before / jpeg_size

        self.psnr = self._psnr_fn(
            transforms.ToTensor()(img_before).unsqueeze(0),
            transforms.ToTensor()(img_after).unsqueeze(0),
        )

        self.psnr_jpeg = self._psnr_fn(
            transforms.ToTensor()(img_before).unsqueeze(0),
            transforms.ToTensor()(jpeg_img).unsqueeze(0),
        )

    def print_summary(self):
        print("\n" + "=" * 30)
        print(f"{self.img_name}: {self.dims[0]}x{self.dims[1]}")
        print(
            f"{self.bpp:>0.2f} bpp | {self.size_before // 1024} KB -> {self.size_after // 1024} KB"
        )
        print(f"{self.ratio:>0.2f}x (jpeg: {self.jpeg_ratio:>0.2f}x)")
        print(f"PSNR: {self.psnr:>0.2f} (jpeg: {self.psnr_jpeg:>0.2f})")
        print("=" * 30)


def eval_compression(model, evaluation_name, datamodule):
    assert datamodule.batch_size == 1  # prevent too big sizes in memmory

    datamodule.setup()
    val_loader = datamodule.val_dataloader()

    print("Running evaluation of compression...\n")

    device = next(model.parameters()).device

    img_comp_metrics_ours = ImageComparisonMetrics("original", "compressed (ours)")
    img_comp_metrics_jpeg = ImageComparisonMetrics("original", "compressed (jpeg)")
    img_patcher = ImagePatcher(patch_size=128)

    compression_metrics = []

    N_IMAGES = 30
    N_SAVE = 5
    for i, batch_tensor in enumerate(val_loader):
        if i + 1 > N_IMAGES:
            break

        if datamodule.ycbcr:
            img = TF.to_pil_image(batch_tensor[0], mode="YCbCr")
        else:
            img = TF.to_pil_image(batch_tensor[0], mode="RGB")

        patches = img_patcher.create_patches(img)

        transform = transforms.ToTensor()
        patches_batch = torch.stack([transform(patch) for _, patch in patches]).to(
            device
        )

        with torch.no_grad():
            reconstructions, bottleneck = model.forward_get_latent(patches_batch)

        if datamodule.ycbcr:
            reconstructions = torch.stack(
                [
                    TF.to_tensor(TF.to_pil_image(img, mode="YCbCr").convert("RGB"))
                    for img in reconstructions
                ]
            )

        # at this point work with rgb original
        img = img.convert("RGB")

        reconstructed = img_patcher.combine_patches(
            img.size,
            [(x, y) for (x, y), _ in patches],
            [transforms.ToPILImage()(r.cpu()) for r in reconstructions],
        )

        img_comp_metrics_ours.update(
            transform(img).unsqueeze(0), transform(reconstructed).unsqueeze(0)
        )
        jpeg_img, _ = get_jpeg_image(img)
        img_comp_metrics_jpeg.update(
            transform(img).unsqueeze(0), transform(jpeg_img).unsqueeze(0)
        )

        if i < N_SAVE:
            compression_metrics.append(
                ImageCompressionMetric(f"img_{i}", img, bottleneck, reconstructed)
            )
            reconstructed.save(f"{OUTPUT_DIR}/{evaluation_name}_{i}_reconstructed.png")

    for metric in compression_metrics:
        metric.print_summary()

    print("\nOur comparison metrics:")
    img_comp_metrics_ours.print_summary()
    print("\nJPEG comparison metrics:")
    img_comp_metrics_jpeg.print_summary()


def eval_patches(model, evaluation_name, datamodule):
    datamodule.setup()
    val_loader = datamodule.val_dataloader()

    print("Running evaluation by patches... \n")

    metrics_compressed = ImageComparisonMetrics("original", "recovered (compressed)")
    metrics_just_cae = ImageComparisonMetrics("original", "recovered (no compression)")

    first_batch_originals = torch.empty(0)
    first_batch_recs_compressed = torch.empty(0)
    first_batch_recs_just_cae = torch.empty(0)

    device = next(model.parameters()).device

    print("Evaluating model over the validation set...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 50:
                break

            originals = batch.to(device)
            recs_compress = model(originals)
            recs_just_cae = model.forward_just_cae(originals)

            if datamodule.ycbcr:
                originals = torch.stack(
                    [
                        TF.to_tensor(TF.to_pil_image(img, mode="YCbCr").convert("RGB"))
                        for img in originals
                    ]
                )
                recs_compress = torch.stack(
                    [
                        TF.to_tensor(TF.to_pil_image(img, mode="YCbCr").convert("RGB"))
                        for img in recs_compress
                    ]
                )
                recs_just_cae = torch.stack(
                    [
                        TF.to_tensor(TF.to_pil_image(img, mode="YCbCr").convert("RGB"))
                        for img in recs_just_cae
                    ]
                )

            metrics_compressed.update(recs_compress, originals)
            metrics_just_cae.update(recs_just_cae, originals)

            if i == 0:
                first_batch_originals = originals
                first_batch_recs_compressed = recs_compress
                first_batch_recs_just_cae = recs_just_cae

    metrics_just_cae.print_summary()
    metrics_compressed.print_summary()

    comparison = torch.cat(
        [first_batch_originals, first_batch_recs_just_cae, first_batch_recs_compressed]
    )

    os.makedirs("outputs", exist_ok=True)
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


def eval_pareto(model, evaluation_name, datamodule):
    assert datamodule.batch_size == 1

    datamodule.setup()
    val_loader = datamodule.val_dataloader()

    print("Running Pareto evaluation...\n")

    device = next(model.parameters()).device

    N_IMAGES = 30
    q_bits_list = [2, 3, 4, 5, 6, 7, 8]

    bpps = []
    psnrs = []
    ssims = []

    for q_bits in q_bits_list:
        model.quantization_bits = q_bits
        print(f"Evaluating with quantization_bits={q_bits}...")

        img_comp_metrics_ours = ImageComparisonMetrics("original", f"ours q={q_bits}")
        img_patcher = ImagePatcher(patch_size=128)

        total_bpp = 0.0
        bpp_count = 0

        for i, batch_tensor in enumerate(val_loader):
            if i + 1 > N_IMAGES:
                break

            if datamodule.ycbcr:
                img = TF.to_pil_image(batch_tensor[0], mode="YCbCr")
            else:
                img = TF.to_pil_image(batch_tensor[0], mode="RGB")

            patches = img_patcher.create_patches(img)
            transform = transforms.ToTensor()
            patches_batch = torch.stack([transform(patch) for _, patch in patches]).to(
                device
            )

            with torch.no_grad():
                reconstructions, bottleneck = model.forward_get_latent(patches_batch)

            if datamodule.ycbcr:
                reconstructions = torch.stack(
                    [
                        TF.to_tensor(
                            TF.to_pil_image(r_img, mode="YCbCr").convert("RGB")
                        )
                        for r_img in reconstructions
                    ]
                )

            img = img.convert("RGB")
            reconstructed = img_patcher.combine_patches(
                img.size,
                [(x, y) for (x, y), _ in patches],
                [transforms.ToPILImage()(r.cpu()) for r in reconstructions],
            )

            img_comp_metrics_ours.update(
                transform(img).unsqueeze(0), transform(reconstructed).unsqueeze(0)
            )

            size_after = len(bottleneck)
            bpp = (size_after * 8.0) / (img.size[0] * img.size[1])
            total_bpp += bpp
            bpp_count += 1

        img_comp_metrics_ours.finilize()
        avg_bpp = total_bpp / bpp_count
        avg_psnr = img_comp_metrics_ours.avg_psnr
        avg_ssim = img_comp_metrics_ours.avg_ssim

        print(
            f"q_bits={q_bits}: BPP={avg_bpp:.4f}, PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}"
        )

        bpps.append(avg_bpp)
        psnrs.append(avg_psnr)
        ssims.append(avg_ssim)

    print("Running JPEG baseline...")
    jpeg_qualities = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    jpeg_bpps, jpeg_psnrs, jpeg_ssims = [], [], []
    for q in jpeg_qualities:
        metrics = ImageComparisonMetrics("original", f"JPEG q={q}")
        total_bpp = 0.0
        count = 0
        for i, batch_tensor in enumerate(val_loader):
            if i + 1 > N_IMAGES:
                break
            if datamodule.ycbcr:
                img = TF.to_pil_image(batch_tensor[0], mode="YCbCr").convert("RGB")
            else:
                img = TF.to_pil_image(batch_tensor[0], mode="RGB")

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            size_after = buf.tell()
            buf.seek(0)
            reconstructed = Image.open(buf)

            transform = transforms.ToTensor()
            metrics.update(
                transform(img).unsqueeze(0), transform(reconstructed).unsqueeze(0)
            )
            total_bpp += (size_after * 8.0) / (img.size[0] * img.size[1])
            count += 1

        metrics.finilize()
        jpeg_bpps.append(total_bpp / count)
        jpeg_psnrs.append(metrics.avg_psnr)
        jpeg_ssims.append(metrics.avg_ssim)

    print("Running JPEG2000 baseline...")
    j2k_rates = [10, 20, 30, 40, 60, 80, 100]
    j2k_bpps, j2k_psnrs, j2k_ssims = [], [], []
    for q in j2k_rates:
        metrics = ImageComparisonMetrics("original", f"JPEG2000 rate={q}")
        total_bpp = 0.0
        count = 0
        for i, batch_tensor in enumerate(val_loader):
            if i + 1 > N_IMAGES:
                break
            if datamodule.ycbcr:
                img = TF.to_pil_image(batch_tensor[0], mode="YCbCr").convert("RGB")
            else:
                img = TF.to_pil_image(batch_tensor[0], mode="RGB")

            buf = io.BytesIO()
            img.save(buf, format="JPEG2000", quality_mode="rates", quality_layers=[q])
            size_after = buf.tell()
            buf.seek(0)
            reconstructed = Image.open(buf)

            transform = transforms.ToTensor()
            metrics.update(
                transform(img).unsqueeze(0), transform(reconstructed).unsqueeze(0)
            )
            total_bpp += (size_after * 8.0) / (img.size[0] * img.size[1])
            count += 1

        metrics.finilize()
        j2k_bpps.append(total_bpp / count)
        j2k_psnrs.append(metrics.avg_psnr)
        j2k_ssims.append(metrics.avg_ssim)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # PSNR plot
    plt.figure()
    plt.plot(bpps, psnrs, marker="o", linestyle="-", color="b", label="Proposed")
    plt.plot(jpeg_bpps, jpeg_psnrs, marker="s", linestyle="--", color="g", label="JPEG")
    plt.plot(
        j2k_bpps, j2k_psnrs, marker="^", linestyle="-.", color="r", label="JPEG2000"
    )
    plt.xlabel("Bits per Pixel (BPP)")
    plt.ylabel("PSNR (dB)")
    plt.title("Pareto Front: PSNR vs BPP")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/{evaluation_name}_pareto_psnr.png")
    plt.close()

    # SSIM plot
    plt.figure()
    plt.plot(bpps, ssims, marker="o", linestyle="-", color="b", label="Proposed")
    plt.plot(jpeg_bpps, jpeg_ssims, marker="s", linestyle="--", color="g", label="JPEG")
    plt.plot(
        j2k_bpps, j2k_ssims, marker="^", linestyle="-.", color="r", label="JPEG2000"
    )
    plt.xlabel("Bits per Pixel (BPP)")
    plt.ylabel("SSIM")
    plt.title("Pareto Front: SSIM vs BPP")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/{evaluation_name}_pareto_ssim.png")
    plt.close()

    print(f"Pareto plots saved to {OUTPUT_DIR}/")


def main():
    os.makedirs("outputs", exist_ok=True)

    # TODO get rid of need for crop datamodule in eval_patches

    # basic_model = load_model_from_checkpoint("basic", "checkpoints/basic_imagenet10k-basic-best-v1.ckpt")

    if False:
        basic_model = torch.load("checkpoints/manual/basic_best.pt", weights_only=False)
        eval_patches(basic_model, "basic_eval", datamodule_imagenet10k_crop)
        eval_compression(basic_model, "basic_eval", datamodule_imagenet10k_no_crop)
        eval_pareto(basic_model, "basic_eval", datamodule_imagenet10k_no_crop)
    else:
        dcal_model = torch.load(
            "checkpoints/manual/DCAL_2018_best_50epoch.pt", weights_only=False
        )
        eval_patches(dcal_model, "basic_eval", datamodule_imagenet10k_crop)
        eval_compression(dcal_model, "basic_eval", datamodule_imagenet10k_no_crop)
        eval_pareto(dcal_model, "basic_eval", datamodule_imagenet10k_no_crop)

    # eval_patches("DCAL_2018", "checkpoints/dcal_combined-DCAL_2018-best.ckpt", datamodule_default_concat)
    # eval_compression("DCAL_2018", "checkpoints/dcal_combined-DCAL_2018-best.ckpt", datamodule_no_crop_concat)


if __name__ == "__main__":
    main()
