import os
import random
from typing import Any, Tuple

from PIL import Image
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
import albumentations as A
import albumentations.pytorch as AP

DATASET_NORMALIZE_INFO = {
    "CIFAR10": {"MEAN": (0.4914, 0.4822, 0.4465), "STD": (0.2470, 0.2435, 0.2616)},
    "CIFAR100": {"MEAN": (0.5071, 0.4865, 0.4409), "STD": (0.2673, 0.2564, 0.2762)},
    "IMAGENET": {"MEAN": (0.485, 0.456, 0.406), "STD": (0.229, 0.224, 0.225)},
    "TACO": {"MEAN": (0.485, 0.456, 0.406), "STD": (0.229, 0.224, 0.225)},
}

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CustomNoisyStudentTrainingDataset(Dataset):
    def __init__(self,root='/opt/ml/data/test/NoLabel'):
        super(CustomNoisyStudentTrainingDataset,self).__init__()
        ## hard augmented
        self.loader = default_loader
        self.samples = [ os.path.join(root,fname) for fname in os.listdir(root) ]
        self.pre = [
             A.PadIfNeeded(min_height=128, min_width=128, p=1),
             A.Resize(224,224)]
        self.post = [
            A.Normalize(
                mean=DATASET_NORMALIZE_INFO["TACO"]["MEAN"],
                std=DATASET_NORMALIZE_INFO["TACO"]["STD"]),
            AP.ToTensorV2()]
        self.transforms_list = [
            A.Rotate(180),
            A.ColorJitter(0.9,0.9,0.9,0.9),
            A.Equalize(),
            A.Affine(
                scale=[0.5,1.5],
                shear=[-30,30],
                translate_percent=[-0.3,0.3]),
            A.Flip(),
            A.GaussianBlur(),
            A.InvertImg(),
            A.ChannelDropout(),
            A.ChannelShuffle(),
            A.RandomGridShuffle(),
            A.RGBShift(r_shift_limit=(-100, 100), g_shift_limit=(-100, 100), b_shift_limit=(-100, 100)),
            A.JpegCompression(quality_lower=50, quality_upper=100),
            A.MultiplicativeNoise(multiplier=(0.8, 1.2), per_channel=True, elementwise=True),
            A.GridDropout()
        ]

    def __len__(self)->int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        sample = np.array(self.loader(path))
        
        selected_transform = random.sample(self.transforms_list,k=2)

        aug_transform = A.Compose( self.pre+ selected_transform + self.post)
        normal_transform = A.Compose(self.pre + self.post)

        aug_sample = aug_transform(image=sample.copy())['image']
        sample = normal_transform(image=sample)['image']
    
        return sample, aug_sample


