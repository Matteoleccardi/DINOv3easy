import os, sys
import argparse

import numpy
from PIL import Image
import torch

import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

from dinov3easy.load.settings import AVAILABLE_MODEL_CHECKPOINTS_PATHS_DICT

from dinov3easy.view._utils import make_transform, TKINTER_ALLOWED_IMAGE_FILES
from dinov3easy.load.load import load_dinov3_models
from dinov3easy.utils.predict import get_features
from dinov3easy.utils.pca import pca_of_features



def run(model_type: str = 'dinov3_vits16', img_resize: int = 1024, img_path: str|None = None):

    if img_path is None:
        root = tk.Tk()
        root.withdraw()
        img_path = filedialog.askopenfilename(
            initialdir=os.path.expanduser("~"),
            title="Select an image file (2d)",
            filetypes=TKINTER_ALLOWED_IMAGE_FILES
        )
        if not img_path:
            print("No image selected, exiting.")
            sys.exit(0)

    img = Image.open(img_path).convert("RGB")

    img_ = numpy.array(img).transpose(2, 0, 1) / 255.0
    mean_ = tuple(img_.mean(axis=(1, 2), keepdims=True).flatten().tolist())
    std_ = tuple(img_.std(axis=(1, 2), keepdims=True).flatten().tolist())

    transform = make_transform(img_resize)
    img: torch.Tensor = transform(img).unsqueeze(0)

    # Load model

    model = load_dinov3_models(model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    with torch.inference_mode():
        img = img.to(device)
        features = get_features(model, img)
    
    # PCA components visualization

    pca_result = pca_of_features(features, n_components=20, more_contrast=True) # -> shape: (1, n_components, H, W)
    pca_result = pca_result.squeeze(0).transpose(1, 2, 0)
    pca_result = (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())
    plt.figure()
    plt.imshow(pca_result[..., :3])
    plt.axis('off')
    plt.title(f"{model_type}, {img_resize}x{img_resize}")
    plt.tight_layout()
    plt.show()

    # Explore each component on its own in a grid
    n_components = pca_result.shape[2]
    fig, axes = plt.subplots(4, 5, figsize=(15, 10))
    for i in range(n_components):
        ax = axes[i // 5, i % 5]
        im = ax.imshow(pca_result[..., i], cmap='hot')
        ax.set_title(f"{i}", fontsize=8)
        ax.axis('off')
    plt.suptitle(f"PCA decomposition (top 3 components) of image features.\n{model_type}, {img_resize}x{img_resize}", fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Visualize DINOv3 features PCA-decomposed.")
    parser.add_argument("--model", type=str, default="dinov3_vits16", choices=AVAILABLE_MODEL_CHECKPOINTS_PATHS_DICT.keys(), help="Type of DINOv3 model to use.")
    parser.add_argument("--img_resize", type=int, default=1024, help="Size to resize images to.")
    args = parser.parse_args()

    run(model_type=args.model, img_resize=args.img_resize)