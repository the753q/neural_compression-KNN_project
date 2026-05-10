import io
import torch
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import kornia

def get_jpeg_image(img, quality=95):
    """
    Returns a JPEG-compressed version of a PIL Image and its size in bytes.
    """
    buf_jpeg = io.BytesIO()
    img.save(buf_jpeg, format="JPEG", quality=quality)
    jpeg_size = buf_jpeg.tell()
    buf_jpeg.seek(0)
    jpeg_img = Image.open(buf_jpeg)
    return jpeg_img, jpeg_size

class ImagePatcher:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def create_patches(self, img_tensor):
        """
        Creates non-overlapping patches from a tensor (C, H, W).
        Returns:
            patches (Tensor): (N, C, patch_size, patch_size)
            positions (list): list of (x, y) coordinates
            original_size (tuple): (width, height)
        """
        C, H, W = img_tensor.shape
        patch_size = self.patch_size

        # Pad to create non-overlapping patches
        pad_w = (patch_size - W % patch_size) % patch_size
        pad_h = (patch_size - H % patch_size) % patch_size
        
        # padding is (left, top, right, bottom)
        padded_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h))
        
        patches = []
        positions = []
        
        _, new_H, new_W = padded_tensor.shape
        for y in range(0, new_H, patch_size):
            for x in range(0, new_W, patch_size):
                patch = padded_tensor[:, y:y+patch_size, x:x+patch_size]
                patches.append(patch)
                positions.append((x, y))
                
        return torch.stack(patches), positions, (W, H)

    def combine_patches(self, img_size, positions, patches):
        """
        Reconstructs an image from patches.
        img_size: (width, height)
        positions: list of (x, y)
        patches: tensor (N, C, patch_size, patch_size)
        """
        W, H = img_size
        C = patches.shape[1]
        reconstructed = torch.zeros((C, H + self.patch_size, W + self.patch_size), device=patches.device)
        
        for (x, y), patch in zip(positions, patches):
            reconstructed[:, y:y+self.patch_size, x:x+self.patch_size] = patch
            
        return reconstructed[:, :H, :W]

def rgb_to_ycbcr(tensor):
    """
    Converts an RGB tensor to YCbCr using Kornia.
    Handles both (C, H, W) and (B, C, H, W).
    """
    if tensor.dim() == 3:
        return kornia.color.rgb_to_ycbcr(tensor.unsqueeze(0)).squeeze(0)
    return kornia.color.rgb_to_ycbcr(tensor)

def ycbcr_to_rgb(tensor):
    """
    Converts a YCbCr tensor to RGB using Kornia.
    Handles both (C, H, W) and (B, C, H, W).
    """
    if tensor.dim() == 3:
        return kornia.color.ycbcr_to_rgb(tensor.unsqueeze(0)).squeeze(0)
    return kornia.color.ycbcr_to_rgb(tensor)

def rgb_to_lab(tensor):
    """
    Converts an RGB tensor to LAB using Kornia.
    Handles both (C, H, W) and (B, C, H, W).
    """
    if tensor.dim() == 3:
        return kornia.color.rgb_to_lab(tensor.unsqueeze(0)).squeeze(0)
    return kornia.color.rgb_to_lab(tensor)

def lab_to_rgb(tensor):
    """
    Converts a LAB tensor to RGB using Kornia.
    Handles both (C, H, W) and (B, C, H, W).
    """
    if tensor.dim() == 3:
        return kornia.color.lab_to_rgb(tensor.unsqueeze(0)).squeeze(0)
    return kornia.color.lab_to_rgb(tensor)

def rgb_to_lab_norm(tensor):
    """
    Converts RGB [0, 1] to LAB [0, 1] normalized.
    L: [0, 100] -> [0, 1]
    a, b: [-128, 127] -> [0, 1]
    """
    lab = rgb_to_lab(tensor)
    l, a, b = torch.split(lab, 1, dim=-3)
    l = l / 100.0
    a = (a + 128.0) / 255.0
    b = (b + 128.0) / 255.0
    return torch.cat([l, a, b], dim=-3)

def lab_norm_to_rgb(tensor):
    """
    Converts LAB [0, 1] normalized to RGB [0, 1].
    """
    l, a, b = torch.split(tensor, 1, dim=-3)
    l = l * 100.0
    a = a * 255.0 - 128.0
    b = b * 255.0 - 128.0
    lab = torch.cat([l, a, b], dim=-3)
    return lab_to_rgb(lab)
