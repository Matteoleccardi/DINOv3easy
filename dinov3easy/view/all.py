import os, sys
import argparse

import numpy
from PIL import Image
import torch

import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

from dinov3easy.load.settings import AVAILABLE_MODEL_CHECKPOINTS_PATHS_DICT
from dinov3easy.view._utils import TKINTER_ALLOWED_IMAGE_FILES

from dinov3easy.view.interactive_attention import run as run_ia
from dinov3easy.view.interactive_pca import run as run_pca
from dinov3easy.view.interactive_similarity import run as run_sim
from dinov3easy.view.interactive_class_similarity import run as run_cs

def run(model_type: str = 'dinov3_vits16', img_resize: int = 1024, img_path: str|None = None):
    # Run all the DINOv3 visualizations in one go, with the same image
    run_pca(model_type=model_type, img_resize=img_resize, img_path=img_path)
    run_ia(model_type=model_type, img_resize=img_resize, img_path=img_path)
    run_sim(model_type=model_type, img_resize=img_resize, img_path=img_path)
    run_cs(model_type=model_type, img_resize=img_resize, img_path=img_path)

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Visualize DINOv3 features PCA-decomposed.")
    parser.add_argument("--model", type=str, default="dinov3_vits16", choices=AVAILABLE_MODEL_CHECKPOINTS_PATHS_DICT.keys(), help="Type of DINOv3 model to use.")
    parser.add_argument("--img_resize", type=int, default=1024, help="Size to resize images to.")
    args = parser.parse_args()

    # Load one image
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

    run(model_type=args.model, img_resize=args.img_resize, img_path=img_path)
