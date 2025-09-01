import torch
from torchvision import transforms

from dinov3easy.utils.constants import (
    IMAGENET_MEAN,
    IMAGENET_STD
)

def make_transform(resize_size: int | list[int] = 768, mean: list|tuple = IMAGENET_MEAN, std: list|tuple = IMAGENET_STD):
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=mean,
        std=std,
    )
    return transforms.Compose([to_tensor, resize, normalize])
