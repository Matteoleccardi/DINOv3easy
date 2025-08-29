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
        # To make this all work, I had to modify the source code of the Transformer, 
        # specifically the dinov3 -> layers -> attention.py file
        # Now, we just have to tell the correct module to save the attention map, complete a forward pass,
        # and then simply retrieve it (use self.save_attention_map, self.attn_map)
        model.blocks[-1].attn.save_attention_map = True
        _ = model(img)
        attn_map = model.blocks[-1].attn.attn_map
        attn_map = attn_map.detach().cpu().numpy()
        attn_map = numpy.mean(attn_map.squeeze(0), axis=0) # shape: (num_tokens, num_tokens)
        # The Dino ViT has 5 extra tokens at the beginning that do not come from image patches
        # (first one is the CLS token, the remaining ones i do not know)
        attn_map = attn_map[5:, 5:] # shape: (num_patches, num_patches)


    # Interactive attention map
    patch_size = model.patch_size
    # - image
    img_show = img[0].permute(1, 2, 0).cpu().numpy()
    img_show_min = img_show.min()
    img_show_max = img_show.max()
    img_show = (img_show - img_show_min) / (img_show_max - img_show_min + 1e-5)
    # - attention map
    index_of_interest = attn_map.shape[0] // 2
    attn_map_show = attn_map.copy()
    n_patches = attn_map_show.shape[0]
    attn_map_show = attn_map_show[index_of_interest].reshape(int(n_patches**0.5), int(n_patches**0.5))
    attn_map_show_min = attn_map_show.min()
    attn_map_show_max = attn_map_show.max()
    attn_map_show = (attn_map_show - attn_map_show_min) / (attn_map_show_max - attn_map_show_min + 1e-5)
    #        attn_map_show = torch.sigmoid(2.0*torch.tensor(attn_map_show)).numpy()
    attn_map_show = numpy.kron(attn_map_show, numpy.ones((patch_size, patch_size))) # resample to the same size as original image (no interpolation)
    # - the patch of interest
    patch_coordinates = numpy.array([index_of_interest // int(n_patches**0.5), index_of_interest % int(n_patches**0.5)])
    patch_coordinates = patch_size*patch_coordinates + patch_size//2
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img_show)
    attn_map_artist = ax.imshow(attn_map_show, cmap='YlOrBr', alpha=0.5)
    patch_position_artist = ax.scatter(patch_coordinates[1], patch_coordinates[0], color='red', s=10)
    # draw a light grid on the image to show where the patches are
    for i in range(int(n_patches**0.5)):
        ax.axhline(y=patch_size*i, color='lightgray', linewidth=0.3)
        ax.axvline(x=patch_size*i, color='lightgray', linewidth=0.3)

    ax.axis('off')
    ax.set_title(f"{model_type}, {img_size}x{img_size}")

    # attach an event listener to the plot, so that when a point is clicked, the attention map overlay changes
    # considering that point
    def on_click(event):
        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            # find the patch that was clicked
            patch_x = int(x // patch_size)
            patch_y = int(y // patch_size)
            # patches are stored as a sequence, so find the index corresponding to the correct patch
            patch_index = patch_y * int(n_patches**0.5) + patch_x
            # update the attention map overlay
            attn_map_show = attn_map.copy()
            attn_map_show = attn_map_show[patch_index].reshape(int(n_patches**0.5), int(n_patches**0.5))
            attn_map_show_min = attn_map_show.min()
            attn_map_show_max = attn_map_show.max()
            attn_map_show = (attn_map_show - attn_map_show_min) / (attn_map_show_max - attn_map_show_min + 1e-5)
            attn_map_show = numpy.kron(attn_map_show, numpy.ones((patch_size, patch_size)))
            attn_map_artist.set_data(attn_map_show)
            # update the scatter point
            patch_coordinates = numpy.array([patch_index // int(n_patches**0.5), patch_index % int(n_patches**0.5)])
            patch_coordinates = patch_size*patch_coordinates + patch_size//2
            patch_position_artist.set_offsets([patch_coordinates[1], patch_coordinates[0]])
            plt.draw()


    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()