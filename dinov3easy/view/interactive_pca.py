import os, sys
import argparse
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from sklearn.decomposition import PCA
import torch
from torchvision import transforms
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


def get_img(paths, transform):
    image_paths = []
    for path in paths:
        image_paths += glob(os.path.join(path, "*.png"))

    images = []
    for image_path in image_paths:
        img = Image.open(image_path).convert("RGB")
        img = transform(img)
        images.append(img)
    images = torch.stack(images, dim=0)  # Stack images into a single tensor
    return images


def create_pca_image(features, n_components=20, H=None, W=None, return_all_components: bool = False, components_in_rgb: list[int]=[0,1,2]):
    N, S, D = features.shape
    if S < n_components:
        raise ValueError(f"Number of tokens {S} is less than the number of PCA components {n_components}.")
    # logic
    features = features.cpu().numpy()
    pca = PCA(n_components=n_components, whiten=True)
    pca_result = pca.fit_transform(features.reshape(N*S, D)) # pca_results shape: (n_patches, components)
    pca_result = torch.sigmoid(torch.tensor(pca_result)*2.0).numpy()
    if H is None or W is None:
        H = W = int(S**0.5)
    pca_result = pca_result.reshape(N, H, W, n_components)
    if return_all_components:
        return pca_result
    else:
        if not isinstance(components_in_rgb, list) or len(components_in_rgb) != 3:
            raise ValueError("components_in_rgb must be a list of three integers.")
        return pca_result[..., components_in_rgb]


# Logic

def main():
    parser = argparse.ArgumentParser(description="Visualize DINOv3 features.")
    parser.add_argument("--model", type=str, default="dinov3_vits16", choices=MODEL_CHECKPOINTS_PATHS_DICT.keys(), help="Type of DINOv3 model to use.")
    parser.add_argument("--img_size", type=int, default=1024, help="Size to resize images to.")
    args = parser.parse_args()
    
    img_size = args.img_size
    model_type = args.model

    # One image at a time
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

    img_ = np.array(img).transpose(2, 0, 1) / 255.0
    mean_ = tuple(img_.mean(axis=(1, 2), keepdims=True).flatten().tolist())
    std_ = tuple(img_.std(axis=(1, 2), keepdims=True).flatten().tolist())

    transform = make_transform(img_size, mean_, std_)
    img: torch.Tensor = transform(img).unsqueeze(0)

    model = load_dinov3_models(model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()
    img = img.to(device)

    with torch.inference_mode():
        token_features = model.get_intermediate_layers(img)[0]
        cls_features = model(img)
        # get the attention map of the first layer
        ################################################   to do
        # Shapes explained:
        # img: (1, 3, img_size, img_size) - input image tensor
        # token_features: (1, num_tokens, embed_dim) - token features from the model
        # cls_features: (1, embed_dim) - class features from the model
        print("Model input shape:", img.shape)
        print("Token features shape:", token_features.shape)
        print("Class features shape:", cls_features.shape)


    # PCA components visualization

    pca_result = create_pca_image(token_features)
    pca_result = pca_result.squeeze(0)
    pca_result = (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())
    plt.figure()
    plt.imshow(pca_result)
    plt.axis('off')
    plt.title(f"{model_type}, {img_size}x{img_size}")
    plt.tight_layout()
    #plt.savefig(os.path.join(BMIC_DATA_PATH, f"pca_result_{model_type}_{img_size}.png"), bbox_inches='tight', pad_inches=0)
    #plt.close()
    plt.show()

    # Explore each component on its own in a grid
    pca_components = create_pca_image(token_features, return_all_components=True)
    n_components = pca_components.shape[3]
    fig, axes = plt.subplots(4, 5, figsize=(15, 10))
    for i in range(n_components):
        ax = axes[i // 5, i % 5]
        im = ax.imshow(pca_components[0, :, :, i], cmap='hot')
        ax.set_title(f"{i}", fontsize=8)
        ax.axis('off')
    plt.suptitle(f"{model_type}, {img_size}x{img_size}", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


    # Interactive attention map



if __name__ == "__main__":
    main()