import os
import requests


WEB_HUB_BASE_URL = "https://example.org/link/to/folder/"

DOWNLOAD_URLS = [
    "dinov3_vits16_pretrain.pth",
    "dinov3_vits16plus_pretrain.pth",
    "dinov3_vitb16_pretrain.pth",
    "dinov3_vitl16_pretrain.pth",
    "dinov3_vith16plus_pretrain.pth",
    "dinov3_vit7b16_pretrain.pth",
    "dinov3_convnext_tiny_pretrain.pth",
    "dinov3_convnext_small_pretrain.pth",
    "dinov3_convnext_base_pretrain.pth",
    "dinov3_convnext_large_pretrain.pth",
    "dinov3_vitl16_pretrain_sat.pth",
    "dinov3_vit7b16_pretrain_sat.pth"
]

TARGET_FOLDER = os.path.join(
    os.path.dirname(__file__), 
    "..", 
    "checkpoints"
)

def download_checkpoints():
    print("Preparing download. This might take a while.")
    print("Be sure to have 65 GB of free disk space.")
    os.makedirs(TARGET_FOLDER, exist_ok=True)
    for filename in DOWNLOAD_URLS:
        url = WEB_HUB_BASE_URL + filename
        target_path = os.path.join(TARGET_FOLDER, filename)
        if os.path.exists(target_path):
            if input(f"{filename} already exists, skip? (y/n)").lower() == 'y':
                continue
        print(f"Downloading {filename}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(target_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"Downloaded {filename} to {TARGET_FOLDER}")
    print("All downloads completed.")
    print(f"Go to:\n{TARGET_FOLDER}\n and check the downloaded files.")

if __name__ == "__main__":
    download_checkpoints()