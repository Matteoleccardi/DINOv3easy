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



def run(img_path: str|None = None):
    # Parser
    parser = argparse.ArgumentParser(description="Visualize DINOv3 class similarity.")
    parser.add_argument("--model", type=str, default="dinov3_vits16", choices=AVAILABLE_MODEL_CHECKPOINTS_PATHS_DICT.keys(), help="Type of DINOv3 model to use.")
    parser.add_argument("--img_size", type=int, default=1024, help="Size to resize images to.")
    args = parser.parse_args()
    
    img_size = args.img_size
    model_type = args.model

    # One image at a time
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

    transform = make_transform(img_size)
    img: torch.Tensor = transform(img).unsqueeze(0)

    # Load model

    model = load_dinov3_models(model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    with torch.inference_mode():
        img = img.to(device)
        features, out = get_features(model, img, dino_output=True) # (1, C, N, N), (1, C)
        features = features[0].detach().cpu().numpy()
        out = out[0].detach().cpu().numpy()

    # Compute similarity
    features /= numpy.linalg.norm(features, axis=0, keepdims=True)
    out /= numpy.linalg.norm(out)
    out = out[:, None, None] # (C, 1, 1)
    cosine_similarity = numpy.sum(features * out, axis=0)
    patch_size = int(img.shape[-1] / features.shape[-1])
    kernel = numpy.ones((patch_size, patch_size))
    cosine_similarity = numpy.kron(cosine_similarity, kernel)
    
    # Visualization

    image_to_show = img.detach().cpu().numpy()[0].transpose(1, 2, 0)
    image_to_show = (image_to_show - image_to_show.min()) / (image_to_show.max() - image_to_show.min())
    plt.figure()
    plt.axis('off')
    plt.imshow(image_to_show)
    plt.imshow(cosine_similarity, cmap='jet', alpha=0.5)
    plt.title(f"Cosine Similarity of DINO output with features.")
    plt.tight_layout()
    plt.show()

    

if __name__ == "__main__":
    run()