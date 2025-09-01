import os, sys, time
import argparse

import numpy
from PIL import Image
import torch
from torchvision import transforms

import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Paths

from settings import (
    MODEL_CHECKPOINTS_PATHS_DICT,
    TK_DEFAULT_IMG_DIR
)

from shared import load_dinov3_models

# Functions

def make_transform(resize_size: int | list[int] = 768, mean: tuple = (0.5, 0.5, 0.5), std: tuple = (0.2, 0.2, 0.2)):
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
    parser = argparse.ArgumentParser(description="Visualize DINOv3 features.")
    parser.add_argument("--model", type=str, default="dinov3_vits16", choices=MODEL_CHECKPOINTS_PATHS_DICT.keys(), help="Type of DINOv3 model to use.")
    parser.add_argument("--img_size", type=int, default=1024, help="Size to resize images to.")
    args = parser.parse_args()
    
    img_size = args.img_size
    model_type = args.model

    # Load one image
    root = tk.Tk()
    root.withdraw()
    img_path = filedialog.askopenfilename(
        initialdir=TK_DEFAULT_IMG_DIR,
        title="Select image file",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    if not img_path:
        sys.exit(0)

    img = Image.open(img_path).convert("RGB")

    img_ = numpy.array(img).transpose(2, 0, 1) / 255.0
    mean_ = tuple(img_.mean(axis=(1, 2), keepdims=True).flatten().tolist())
    std_ = tuple(img_.std(axis=(1, 2), keepdims=True).flatten().tolist())

    transform = make_transform(img_size, mean_, std_)
    img: torch.Tensor = transform(img).unsqueeze(0) # shape: (1, 3, img_size, img_size)

    # Load model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dinov3_models(model_type)
    model = model.to(device).eval()


    with torch.inference_mode():
        img = img.to(device)
        token_features = model.get_intermediate_layers(img)[0]
        token_features = token_features.detach().cpu().numpy()
        token_features = token_features.squeeze(0)  # shape: (num_tokens, embed_dim)
        token_features = token_features / numpy.linalg.norm(token_features, axis=-1, keepdims=True)  # normalize to unit norm
        token_features = token_features.reshape(
            int(token_features.shape[0]**0.5),
            int(token_features.shape[0]**0.5),
            -1
        )
        token_features = token_features.transpose(2, 0, 1)  # shape: (embed_dim, K1, K2)

    # Interactive cosine similarity map
    patch_size = model.patch_size
    # - image
    img_show = img[0].permute(1, 2, 0).cpu().numpy()
    img_show_min = img_show.min()
    img_show_max = img_show.max()
    img_show = (img_show - img_show_min) / (img_show_max - img_show_min + 1e-5)
    # - similarity
    index_of_interest = (0, 0)
    def get_similarity_map(features: numpy.ndarray, k1: int = 0, k2: int = 0):
        # features must be of shape (channels, K1, K2)
        # Normalize features
        features = features / numpy.linalg.norm(features, axis=0, keepdims=True)
        # Get query vector to have same shape as features
        query = features[:, k1:k1+1, k2:k2+1]  # (channels, 1, 1)
        # Compute similarity = dot(query, features) / (norm(query) * norm(features))
        cosine_similarity = numpy.sum(features * query, axis=0) / 2.0 # shape: (K1, K2)
        return cosine_similarity
    
    def get_similarity_map_discrete_image(similarity_map: numpy.ndarray):
        # Define similarity intervals (in [0;1] interval, monotone decreasing)
        intervals = [1.0, 0.95, 0.75, 0.60, 0.50, 0.0]
        # Define some RGBA colors, one for each similarity cathegory
        color_1 = numpy.array([1.0, 0.1, 0.1, 0.5])
        color_2 = numpy.array([1.0, 0.5, 0.1, 0.5])
        color_3 = numpy.array([1.0, 1.0, 0.1, 0.4])
        color_4 = numpy.array([0.5, 1.0, 0.1, 0.2])
        color_5 = numpy.array([0.0, 0.0, 0.0, 0.0])
        colors = [color_1, color_2, color_3, color_4, color_5]
        # Check similarity map shape
        assert len(similarity_map.shape) == 2
        assert similarity_map.shape[0] == similarity_map.shape[1]
        # Check similarity map intensities and normalize it
        similarity_map = (similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min() + 1e-5)
        similarity_map = numpy.clip(similarity_map, 0, 1)
        # Create image
        similarity_image = numpy.zeros((*similarity_map.shape, 4), dtype=numpy.float32)
        for int_s, int_e, int_c in zip(intervals[:-1], intervals[1:], colors):
            mask = (similarity_map <= int_s) & (similarity_map > int_e)
            similarity_image[mask] = int_c
        return similarity_image

    similarity_map = get_similarity_map(token_features)
    similarity_map = numpy.kron(similarity_map, numpy.ones((patch_size, patch_size)))
    # - the patch of interest
    patch_coordinates = numpy.array([index_of_interest[0], index_of_interest[1]])
    patch_coordinates = patch_size*patch_coordinates + patch_size//2 + 0.5



    fig = plt.figure()
    # interactive choice and similarity scores
    ax = fig.add_subplot(121)
    ax.imshow(img_show)
    similarity_map_artist = ax.imshow(similarity_map, cmap='YlOrBr', alpha=0.5)
    patch_position_artist = ax.scatter(patch_coordinates[1], patch_coordinates[0], color='red', s=10)
    for i in range(img_show.shape[0] // patch_size + 1):
        ax.axhline(y=patch_size*i, color='lightgray', linewidth=0.5)
        ax.axvline(x=patch_size*i, color='lightgray', linewidth=0.5)
    # dicrete similarity
    ax2 = fig.add_subplot(122)
    ax2.imshow(img_show)
    similarity_map_discrete = get_similarity_map_discrete_image(similarity_map)
    discrete_map_artist = ax2.imshow(similarity_map_discrete, vmin=0.0, vmax=1.0)
    patch_position_artist_ax2 = ax2.scatter(patch_coordinates[1], patch_coordinates[0], color='red', s=10)

    ax.axis('off')
    ax.set_title(f"{model_type}, {img_size}x{img_size}")

    # attach an event listener to the plot, so that when a point is clicked, the map overlay changes
    # considering that point
    def on_click(event):
        if event.inaxes == ax or event.inaxes == ax2:
            x, y = int(event.xdata), int(event.ydata)
            # find the patch that was clicked
            patch_k1 = int(y // patch_size)
            patch_k2 = int(x // patch_size)
            # ax
            # update the similarity map overlay
            similarity_map = get_similarity_map(token_features, patch_k1, patch_k2)
            similarity_map = numpy.kron(similarity_map, numpy.ones((patch_size, patch_size)))
            similarity_map_artist.set_data(similarity_map)
            # update the scatter point
            patch_coordinates = numpy.array([patch_k1, patch_k2])
            patch_coordinates = patch_size*patch_coordinates + patch_size//2 + 0.5
            patch_position_artist.set_offsets([patch_coordinates[1], patch_coordinates[0]])
            plt.draw()
            # ax2
            similarity_map_discrete = get_similarity_map_discrete_image(similarity_map)
            discrete_map_artist.set_data(similarity_map_discrete)
            patch_position_artist_ax2.set_offsets([patch_coordinates[1], patch_coordinates[0]])


    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()