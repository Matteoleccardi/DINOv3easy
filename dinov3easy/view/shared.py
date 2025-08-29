import torch

from settings import DINOV3_REPO_PATH, MODEL_CHECKPOINTS_PATHS_DICT


def load_dinov3_models(model_type):
    # DINOv3 ViT models pretrained on web images
    model = torch.hub.load(
        DINOV3_REPO_PATH, 
        model_type, 
        source='local', 
        weights=MODEL_CHECKPOINTS_PATHS_DICT[model_type]
    )
    return model

