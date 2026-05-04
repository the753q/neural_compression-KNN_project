from pathlib import Path

import lightning.pytorch as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn

class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        sample = self.dataset[index]
        # ImageFolder returns (sample, target), we only care about sample (image)
        if isinstance(sample, tuple):
            sample = sample[0]
            
        if self.transform:
            sample = self.transform(sample)
        return sample, 0 # Return 0 as dummy target

    def __len__(self):
        return len(self.dataset)

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

class DataModuleBase(pl.LightningDataModule):
    def __init__(self, random_crop, ycbcr, batch_size=64, num_workers=4, patch_size=256):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_crop = random_crop
        self.ycbcr = ycbcr
        self.patch_size = patch_size

        # Train transform: with random crop if requested
        t_train = []
        if random_crop:
            t_train.append(transforms.RandomCrop((self.patch_size, self.patch_size)))
        if ycbcr:
            t_train.append(transforms.Lambda(lambda x: x.convert('YCbCr')))
        t_train.append(transforms.ToTensor())
        self.train_transform = transforms.Compose(t_train)

        # Val/Test transform: NO random crop
        t_val = []
        if ycbcr:
            t_val.append(transforms.Lambda(lambda x: x.convert('YCbCr')))
        t_val.append(transforms.ToTensor())
        self.val_transform = transforms.Compose(t_val)
        
        self.collate_fn = lambda batch: torch.stack([img for img, _ in batch])

    def setup(self, x):
        raise NotImplementedError("Subclasses must implement this method.")

    def train_dataloader(self):
        assert hasattr(self, 'train_ds')
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        assert hasattr(self, 'val_ds')
        # If we are not cropping, we must use batch_size=1 for varying image sizes
        bs = self.batch_size if not self.random_crop else 1
        return DataLoader(self.val_ds, batch_size=bs, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloader(self):
        assert hasattr(self, 'test_ds')
        bs = self.batch_size if not self.random_crop else 1
        return DataLoader(self.test_ds, batch_size=bs, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

class ClassImagesDataModule(DataModuleBase):
    def __init__(self, data_dir, random_crop, ycbcr, batch_size=64, num_workers=4, patch_size = 256):
        super().__init__(random_crop = random_crop, ycbcr=ycbcr,
                        batch_size=batch_size, num_workers=num_workers, patch_size=patch_size)

        self.data_dir = data_dir

    def setup(self, stage=None):
        dataset = ImageFolder(self.data_dir, transform=None)

        # Filter images smaller than patch size if cropping is enabled
        if self.random_crop:
            indices = [
                i for i, (path, _) in enumerate(dataset.samples)
                if min(Image.open(path).size) >= self.patch_size
            ]
            dataset = Subset(dataset, indices)

        train_indices, val_indices, test_indices = random_split(
            range(len(dataset)), [0.8, 0.1, 0.1],
            generator=torch.Generator().manual_seed(42)
        )

        self.train_ds = TransformDataset(Subset(dataset, train_indices), transform=self.train_transform)
        self.val_ds   = TransformDataset(Subset(dataset, val_indices),   transform=self.val_transform)
        self.test_ds  = TransformDataset(Subset(dataset, test_indices),  transform=self.val_transform)
    
class DF2KDataModule(DataModuleBase):
    def __init__(self, train_dir, test_dir, random_crop, ycbcr, batch_size=64, num_workers=4, patch_size = 256):
        super().__init__(random_crop = random_crop, ycbcr=ycbcr,
                batch_size=batch_size, num_workers=num_workers, patch_size=patch_size)
        
        self.train_dir = train_dir
        self.test_dir = test_dir

    def setup(self, stage=None):
        train_dataset_raw = ImageFolder(self.train_dir, transform=None)
        test_dataset_raw  = ImageFolder(self.test_dir,   transform=None)

        train_indices, val_indices = random_split(
            range(len(train_dataset_raw)), [0.9, 0.1],
            generator=torch.Generator().manual_seed(42)
        )

        self.train_ds = TransformDataset(Subset(train_dataset_raw, train_indices), transform=self.train_transform)
        self.val_ds   = TransformDataset(Subset(train_dataset_raw, val_indices),   transform=self.val_transform)
        self.test_ds  = TransformDataset(test_dataset_raw, transform=self.val_transform)