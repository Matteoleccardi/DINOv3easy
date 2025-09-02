import os, sys

_this_file_dir = os.path.dirname(__file__)

DINOV3_REPO_PATH = os.path.join(_this_file_dir, "..", "..", "dinov3")



# MODEL_CHECKPOINTS_PATH
_checkpoints_location_file = os.path.join(
        os.path.split(_this_file_dir)[0],
        "checkpoints", "checkpoints_location.txt"
    )
MODEL_CHECKPOINTS_PATH = open(_checkpoints_location_file).read().strip()

if not os.path.isdir(MODEL_CHECKPOINTS_PATH):
    print(f"Warning: The specified MODEL_CHECKPOINTS_PATH does not exist or is not a directory:\n\t{MODEL_CHECKPOINTS_PATH}")
    print("Please run the script dinov3easy/setup/checkpoints.py to set the correct path.")
    sys.exit(1)

_pth_files = [f for f in os.listdir(MODEL_CHECKPOINTS_PATH) if f.endswith(".pth")]
if len(_pth_files) == 0:
    print(f"No .pth files found in the specified MODEL_CHECKPOINTS_PATH:\n\t{MODEL_CHECKPOINTS_PATH}")
    print("Check again their existence, and be sure that the file extension is '.pth'")
    sys.exit(1)


# MODEL_CHECKPOINTS_PATHS_DICT

_pth_sat_files = [f for f in _pth_files if "sat" in f]

def _get_corresponding_file(model_name: str):
    _lookin_list = []
    if "sat" in model_name:
        model_name = model_name.replace("_sat", "")
        _lookin_list = _pth_sat_files
    else:
        _lookin_list = _pth_files

    for f in _lookin_list:
        if model_name+"_" in f:
            return f
    return None

MODEL_CHECKPOINTS_PATHS_DICT = {
    # ViT models (can have attention map)
    'dinov3_vits16':         None,
    'dinov3_vits16plus':     None,
    'dinov3_vitb16':         None,
    'dinov3_vitl16':         None,
    'dinov3_vith16plus':     None,
    'dinov3_vit7b16':        None,
    # ConvNeXt models (no attention map)
    'dinov3_convnext_tiny':  None,
    'dinov3_convnext_small': None,
    'dinov3_convnext_base':  None,
    'dinov3_convnext_large': None,
    # For satellite images
    'dinov3_vitl16_sat':     None,
    'dinov3_vit7b16_sat':    None,
}

for _key in MODEL_CHECKPOINTS_PATHS_DICT.keys():
    MODEL_CHECKPOINTS_PATHS_DICT[_key] = os.path.join(
        MODEL_CHECKPOINTS_PATH,    
        _get_corresponding_file(_key)
    ) if _get_corresponding_file(_key) is not None else None
    # it is allowed for a model checkpoint to not be available

AVAILABLE_MODEL_CHECKPOINTS_PATHS_DICT = {}
for _key, _value in MODEL_CHECKPOINTS_PATHS_DICT.items():
    if _value is not None:
        AVAILABLE_MODEL_CHECKPOINTS_PATHS_DICT[_key] = _value
