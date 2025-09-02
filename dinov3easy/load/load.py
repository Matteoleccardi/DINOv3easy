import torch

from dinov3easy.load.settings import DINOV3_REPO_PATH, MODEL_CHECKPOINTS_PATHS_DICT

def load_dinov3_models(model_type) -> torch.nn.Module:
    """
    loads the dinov3 model of choice (instantiates the class, which is a torch.nn.Module).

    # Input

    `model_type`: (str) The call-sign of the pretrained model to instantiate and load. Possible choices are:

    - ViT models (can have attention map)
      - 'dinov3_vits16'
      - 'dinov3_vits16plus'
      - 'dinov3_vitb16'
      - 'dinov3_vitl16'
      - 'dinov3_vith16plus'
      - 'dinov3_vit7b16'
    - ConvNeXt models (no attention map)
      - 'dinov3_convnext_tiny'
      - 'dinov3_convnext_small'
      - 'dinov3_convnext_base'
      - 'dinov3_convnext_large'
    - ViT for satellite images
      - 'dinov3_vitl16_sat'
      - 'dinov3_vit7b16_sat'

    # Output

    The requested model with loaded weights, which is a torch.nn.Module.
    """
    if model_type not in MODEL_CHECKPOINTS_PATHS_DICT:
        raise ValueError(f"Unknown model type: {model_type}.\nChoose from {list(MODEL_CHECKPOINTS_PATHS_DICT.keys())}.\nIf unsure, start with 'dinov3_vits16'.")
    if MODEL_CHECKPOINTS_PATHS_DICT[model_type] is None:
        raise ValueError(f"No checkpoint available for model type: {model_type}. Check if the related checkpoint is available in the specified checkpoints folder.\n\
                           To setup the correct folder that stores the model checkpoints, run 'python ./dinov3easy/setup/checkpoints.py'.")
    model = torch.hub.load(
        DINOV3_REPO_PATH, 
        model_type, 
        source='local', 
        weights=MODEL_CHECKPOINTS_PATHS_DICT[model_type]
    )
    return model
