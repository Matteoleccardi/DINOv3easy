import os, sys, time
import argparse

import numpy
from PIL import Image
import torch
from torchvision import transforms

import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

from dinov3easy.load.settings import (
    MODEL_CHECKPOINTS_PATHS_DICT
)
from dinov3easy.utils.constants import (
    IMAGENET_MEAN,
    IMAGENET_STD
)

from dinov3easy.load.load import load_dinov3_models
from dinov3easy.utils.predict import get_attention_map
from dinov3easy.view.visualizers import AttentionMapInteractiveVisualizer

# Functions

def make_transform(resize_size: int | list[int] = 768, mean: list|tuple = IMAGENET_MEAN, std: list|tuple = IMAGENET_STD):
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=mean,
        std=std,
    )
    return transforms.Compose([to_tensor, resize, normalize])



# Logic

def main():
    
    # program args
    parser = argparse.ArgumentParser(description="Visualize DINOv3 attention map interactively.")
    parser.add_argument("--model", type=str, default="dinov3_vits16", choices=MODEL_CHECKPOINTS_PATHS_DICT.keys(), help="Type of DINOv3 model to use.")
    parser.add_argument("--img_size", type=int, default=1024, help="Size to resize images to.")
    args = parser.parse_args()
    
    img_size = args.img_size
    model_type = args.model

    # Load one image
    root = tk.Tk()
    root.withdraw()
    img_path = filedialog.askopenfilename(
        initialdir=os.path.expanduser("~"),
        title="Select an image file (2d)",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    if not img_path:
        print("No image selected, exiting.")
        sys.exit(0)

    img = Image.open(img_path).convert("RGB")

    img_ = numpy.array(img).transpose(2, 0, 1) / 255.0
    mean_ = tuple(img_.mean(axis=(1, 2), keepdims=True).flatten().tolist())
    std_ = tuple(img_.std(axis=(1, 2), keepdims=True).flatten().tolist())

    transform = make_transform(img_size)
    img: torch.Tensor = transform(img).unsqueeze(0) # shape: (1, 3, img_size, img_size)

    # Load model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dinov3_models(model_type)
    model = model.to(device).eval()

    with torch.inference_mode():
        img = img.to(device)
        attn_map = get_attention_map(model, img, remove_extra_tokens=True) 
        # shape: (num_patches, num_patches)
        
    
    _ = AttentionMapInteractiveVisualizer(
        image = img_.transpose(1, 2, 0),
        attention_map = attn_map.detach().cpu().numpy()
    )
    plt.show()




if __name__ == "__main__":
    main()