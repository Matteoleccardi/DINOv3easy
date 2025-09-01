import os, sys
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

from dinov3easy.view._utils import make_transform
from dinov3easy.load.load import load_dinov3_models
from dinov3easy.utils.predict import get_features
from dinov3easy.visualizers import SimilarityMapInteractiveVisualizer


def main():
    
    # program args
    parser = argparse.ArgumentParser(description="Visualize DINOv3 features map cosine similarity interactively.")
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
        features = get_features(model, img)[0].detach().cpu().numpy() # (C, K, K)

    # Interactive cosine similarity map
    image_to_show = img.detach().cpu().numpy()[0].transpose(1, 2, 0)
    image_to_show = (image_to_show - image_to_show.min()) / (image_to_show.max() - image_to_show.min())
    _ = SimilarityMapInteractiveVisualizer(
        image = image_to_show,
        features = features
    )
    plt.show()



if __name__ == "__main__":
    main()