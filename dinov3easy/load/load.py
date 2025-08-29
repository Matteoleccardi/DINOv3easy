import torch

from .settings import DINOV3_REPO_PATH, MODEL_CHECKPOINTS_PATHS_DICT

def load_dinov3_models(model_type) -> torch.nn.Module:
    if model_type not in MODEL_CHECKPOINTS_PATHS_DICT:
        raise ValueError(f"Unknown model type: {model_type}.\nChoose from {list(MODEL_CHECKPOINTS_PATHS_DICT.keys())}.\nIf unsure, start with 'dinov3_vits16'.")
    model = torch.hub.load(
        DINOV3_REPO_PATH, 
        model_type, 
        source='local', 
        weights=MODEL_CHECKPOINTS_PATHS_DICT[model_type]
    )
    return model
