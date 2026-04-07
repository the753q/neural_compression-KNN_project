from pathlib import Path

import lightning.pytorch as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def subset_dataset(dataset, limit):
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


class DatasetFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.image_folder = ImageFolder(root=str(self.root))
        self.samples = list(self.image_folder.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, _ = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


class ImageNetSubsetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=8, num_workers=2, image_size=256, train_limit=None, val_limit=None):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.train_limit = train_limit
        self.val_limit = val_limit
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        train_dataset = DatasetFolder(self.train_dir, transform=self.transform)
        val_dataset = DatasetFolder(self.val_dir, transform=self.transform)
        self.train_dataset = subset_dataset(train_dataset, self.train_limit)
        self.val_dataset = subset_dataset(val_dataset, self.val_limit)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
