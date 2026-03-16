import os
from torch.utils.data import Dataset
from PIL import Image
import random   
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

IMG_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((480, 640))
])

MASK_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((480, 640), interpolation=transforms.InterpolationMode.NEAREST)
])

class OrganoidDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None, target=False, train=False):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.transform = transform
        self.target = target
        self.target_transform = target_transform
        self.train = train

        self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg', '.tif', '.tiff'))])
        self.masks = sorted([f for f in os.listdir(self.mask_dir) if f.endswith(('.png', '.jpg', '.tif', '.tiff'))])
        if target:
            self.target_dir = os.path.join(root_dir, "targets")
            self.targets = sorted([f for f in os.listdir(self.target_dir) if f.endswith(('.png', '.jpg', '.tif', '.tiff'))])
            if len(self.targets) == 0:
                print(f"ATTENTION: No targets found in {self.target_dir}")
        
        if len(self.images) == 0:
            print(f"ATTENTION: No images found in {self.img_dir}")
        if len(self.masks) == 0:
            print(f"ATTENTION: No masks found in {self.mask_dir}")
        if len(self.images) != len(self.masks):
            raise ValueError("Number of images and masks do not match!")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        mask_name = self.masks[idx]
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
            
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]

        image = self._normalize_image(idx)

        mask_array = (mask_array > 0).astype(np.float32) 
        
        if self.transform:
            image = self.transform(image)
            if self.target:
                target_array = self._normalize_image(idx)
                target = self.transform(target_array)
        if self.target_transform:
            mask = self.target_transform(mask_array)

        if self.target:
            return image, mask, target
        else:
            if self.train:
                if random.random() > 0.5:
                    image = TF.hflip(image)
                    mask = TF.hflip(mask)
                if random.random() > 0.5:
                    image = TF.vflip(image)
                    mask = TF.vflip(mask)
            return image, mask

    def _normalize_image(self, idx):
        """Helper function to normalize image data to [0, 1] range."""
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)

        img_array = np.array(image)
        if img_array.ndim == 3:
            img_array = np.mean(img_array[..., :3], axis=2) 
        img_array = img_array.astype(np.float32)
        img_min = img_array.min()
        img_max = img_array.max()
        if img_max > img_min:
            image_array = (img_array - img_min) / (img_max - img_min)
        else:
            image_array = img_array - img_min
        return image_array