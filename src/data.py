from pathlib import Path

import lightning.pytorch as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn

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
    def __init__(self, random_crop, ycbcr, batch_size=64, num_workers=4, patch_size=256, val_patch_size=256, val_batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_crop = random_crop
        self.ycbcr = ycbcr
        self.patch_size = patch_size
        self.val_patch_size = val_patch_size
        self.val_batch_size = val_batch_size

        assert not ((not random_crop) and (batch_size > 1)), "Can't combine images of various sizes in one batch."

        t = []
        val_t = []
        if random_crop:
            t.append(transforms.RandomCrop((self.patch_size, self.patch_size)))
            if val_patch_size:
                val_t.append(transforms.CenterCrop((self.val_patch_size, self.val_patch_size)))
        if ycbcr:
            t.append(transforms.Lambda(lambda x: x.convert('YCbCr')))
            val_t.append(transforms.Lambda(lambda x: x.convert('YCbCr')))

        t.append(transforms.ToTensor())
        val_t.append(transforms.ToTensor())

        self.transform = transforms.Compose(t)
        self.val_transform = transforms.Compose(val_t)
        
        self.collate_fn=lambda batch: torch.stack([img for img, _ in batch])

    def setup(self, x):
        raise NotImplementedError("Subclasses must implement this method.")

    def train_dataloader(self):
        assert hasattr(self, 'train_ds')
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        assert hasattr(self, 'val_ds')
        return DataLoader(self.val_ds, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloader(self):
        assert hasattr(self, 'test_ds')
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

class ClassImagesDataModule(DataModuleBase):
    def __init__(self, data_dir, random_crop, ycbcr, batch_size=64, num_workers=4, patch_size = 256):
        super().__init__(random_crop = random_crop, ycbcr=ycbcr,
                        batch_size=batch_size, num_workers=num_workers, patch_size=patch_size,
                        val_batch_size=batch_size, val_patch_size=patch_size)

        self.data_dir = data_dir

    def setup(self, stage=None):
        dataset = ImageFolder(self.data_dir, transform=self.transform)

        dataset = Subset(dataset, [
            i for i, (path, _) in enumerate(dataset.samples)
                if min(Image.open(path).size) >= self.patch_size
        ]) # remove images smaller than patch size

        self.train_ds, self.val_ds, self.test_ds = random_split(
            dataset, [0.8, 0.1, 0.1],
            generator=torch.Generator().manual_seed(42)
        )
    
class DF2KDataModule(DataModuleBase):
    def __init__(self, train_dir, test_dir, random_crop, ycbcr, batch_size=64, num_workers=4,
                  patch_size = 256, val_patch_size = 512, val_batch_size = 16):
        super().__init__(random_crop = random_crop, ycbcr=ycbcr,
                batch_size=batch_size, num_workers=num_workers,
                  patch_size=patch_size, val_patch_size=val_patch_size, val_batch_size=val_batch_size)
        
        self.train_dir = train_dir
        self.test_dir = test_dir

    def setup(self, stage=None):
        train_dataset = ImageFolder(self.train_dir, transform=self.transform)
        val_dataset = ImageFolder(self.train_dir, transform=self.val_transform)
        test_dataset   = ImageFolder(self.test_dir, transform=self.transform)

        self.test_ds = test_dataset

        indices = list(range(len(train_dataset)))
        split = int(0.9 * len(indices))
        rng = torch.Generator().manual_seed(42)
        perm = torch.randperm(len(indices), generator=rng).tolist()

        train_indices = perm[:split]
        val_indices = perm[split:]

        self.train_ds = Subset(train_dataset, train_indices)
        self.val_ds = Subset(val_dataset, val_indices)

class MinecraftDataModule(DataModuleBase):
    def __init__(self, train_dir, test_dir, random_crop, ycbcr, batch_size=64, num_workers=4,
                  patch_size = 256, val_patch_size = 512, val_batch_size = 16):
        super().__init__(random_crop = random_crop, ycbcr=ycbcr,
                batch_size=batch_size, num_workers=num_workers,
                  patch_size=patch_size, val_patch_size=val_patch_size, val_batch_size=val_batch_size)
        
        self.train_dir = train_dir
        self.test_dir = test_dir

    def setup(self, stage=None):
        train_dataset = ImageFolder(self.train_dir, transform=self.transform)
        val_dataset = ImageFolder(self.train_dir, transform=self.val_transform)
        test_dataset   = ImageFolder(self.test_dir, transform=self.transform)

        self.test_ds = test_dataset

        indices = list(range(len(train_dataset)))
        split = int(0.9 * len(indices))
        rng = torch.Generator().manual_seed(42)
        perm = torch.randperm(len(indices), generator=rng).tolist()

        train_indices = perm[:split]
        val_indices = perm[split:]

        self.train_ds = Subset(train_dataset, train_indices)
        self.val_ds = Subset(val_dataset, val_indices)