from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_DIR / "datasets"
COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive")
GOOGLE_DRIVE_DATASETS_DIR = Path("/content/drive/MyDrive/datasets_compression")
IMAGE_SUFFIXES = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")


class SimpleImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        return Image.open(self.image_paths[index]).convert("RGB")


def _collect_image_paths(directory, recursive=False):
    image_paths = []

    for suffix in IMAGE_SUFFIXES:
        if recursive:
            image_paths.extend(directory.rglob(suffix))
        else:
            image_paths.extend(directory.glob(suffix))

    return sorted(image_paths)


def _limit_image_paths(image_paths, max_images=None):
    if max_images is None:
        return image_paths
    return image_paths[:max_images]


def _ensure_directory(directory):
    directory.mkdir(parents=True, exist_ok=True)


def _resolve_datasets_dir(is_colab=False):
    if is_colab:
        if not COLAB_DRIVE_ROOT.exists():
            raise RuntimeError(
                "Google Drive is not mounted. Mount it before calling "
                "retrieve_dataset(..., is_colab=True)."
            )
        return GOOGLE_DRIVE_DATASETS_DIR
    return DATASETS_DIR


def _missing_dataset_message(dataset_name, kaggle_link, expected_path):
    return (
        f"{dataset_name} was not found.\n"
        f"Download it from: {kaggle_link}\n"
        f"Extract it into: {expected_path}"
    )


class Kodak24DataModule(pl.LightningDataModule):
    KAGGLE_LINK = "https://www.kaggle.com/datasets/drxinchengzhu/kodak24"

    def __init__(
        self,
        batch_size=1,
        num_workers=0,
        max_images=None,
        data_dir=None,
        is_colab=False,
    ):
        super().__init__()
        self.datasets_dir = _resolve_datasets_dir(is_colab)
        self.data_dir = (
            Path(data_dir) if data_dir is not None else self.datasets_dir / "Kodak24"
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_images = max_images
        self.dataset = None
        self.missing_message = None

    def prepare_data(self):
        _ensure_directory(self.datasets_dir)
        _ensure_directory(self.data_dir)

        image_paths = _collect_image_paths(self.data_dir)
        if not image_paths:
            self.missing_message = _missing_dataset_message(
                "Kodak24", self.KAGGLE_LINK, self.data_dir
            )
            return False

        self.missing_message = None
        return True

    def setup(self, stage=None):
        image_paths = _collect_image_paths(self.data_dir)
        image_paths = _limit_image_paths(image_paths, self.max_images)
        self.dataset = SimpleImageDataset(image_paths)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class ImageNetDataModule(pl.LightningDataModule):
    KAGGLE_LINK = "https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000"

    def __init__(
        self,
        batch_size=4,
        num_workers=0,
        max_train_images=None,
        max_val_images=None,
        data_dir=None,
        is_colab=False,
    ):
        super().__init__()
        self.datasets_dir = _resolve_datasets_dir(is_colab)
        self.data_dir = (
            Path(data_dir) if data_dir is not None else self.datasets_dir / "ImageNet"
        )
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_train_images = max_train_images
        self.max_val_images = max_val_images
        self.train_dataset = None
        self.val_dataset = None
        self.missing_message = None

    def prepare_data(self):
        _ensure_directory(self.datasets_dir)
        _ensure_directory(self.data_dir)
        _ensure_directory(self.train_dir)
        _ensure_directory(self.val_dir)

        if not _collect_image_paths(self.train_dir, recursive=True):
            self.missing_message = _missing_dataset_message(
                "ImageNet train split", self.KAGGLE_LINK, self.train_dir
            )
            return False

        if not _collect_image_paths(self.val_dir, recursive=True):
            self.missing_message = _missing_dataset_message(
                "ImageNet validation split", self.KAGGLE_LINK, self.val_dir
            )
            return False

        self.missing_message = None
        return True

    def setup(self, stage=None):
        train_images = _limit_image_paths(
            _collect_image_paths(self.train_dir, recursive=True),
            self.max_train_images,
        )
        val_images = _limit_image_paths(
            _collect_image_paths(self.val_dir, recursive=True),
            self.max_val_images,
        )
        self.train_dataset = SimpleImageDataset(train_images)
        self.val_dataset = SimpleImageDataset(val_images)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def retrieve_dataset(
    dataset_name,
    batch_size=1,
    num_workers=0,
    data_dir=None,
    is_colab=False,
    max_images=None,
    max_train_images=None,
    max_val_images=None,
):
    dataset_name = dataset_name.lower()

    if dataset_name == "kodak24":
        datamodule = Kodak24DataModule(
            batch_size=batch_size,
            num_workers=num_workers,
            max_images=max_images,
            data_dir=data_dir,
            is_colab=is_colab,
        )
    elif dataset_name == "imagenet":
        datamodule = ImageNetDataModule(
            batch_size=batch_size,
            num_workers=num_workers,
            max_train_images=max_train_images or max_images,
            max_val_images=max_val_images or max_images,
            data_dir=data_dir,
            is_colab=is_colab,
        )
    else:
        raise ValueError(f"Dataset '{dataset_name}' not implemented")

    dataset_ready = datamodule.prepare_data()
    datamodule.setup()
    if not dataset_ready and datamodule.missing_message is not None:
        print(datamodule.missing_message)
    return datamodule
