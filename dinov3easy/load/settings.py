import os

_this_file_dir = os.path.dirname(__file__)

DINOV3_REPO_PATH = os.path.join(_this_file_dir, "..", "dinov3")



# MODEL_CHECKPOINTS_PATH

_package_checkpoints_path = os.path.join(_this_file_dir, "..", "checkpoints")

_list_of_files = os.listdir(_package_checkpoints_path) if os.path.exists(_package_checkpoints_path) else []
_list_of_files = [l for l in _list_of_files if l.endswith(".pth")]

if len(_list_of_files) != 0:
    MODEL_CHECKPOINTS_PATH = os.path.join(_package_checkpoints_path)
else:
    _location_txt_file_path = os.path.join(_package_checkpoints_path, "checkpoints_location.txt")
    if os.path.exists(_location_txt_file_path):
        _content = open(_location_txt_file_path).read()
        _content = _content.strip()
        _content = str(os.path.normpath(_content))
        if os.path.exists(_content):
            MODEL_CHECKPOINTS_PATH = _content
            _files = os.listdir(MODEL_CHECKPOINTS_PATH)
            _files = [f for f in _files if f.endswith(".pth")]
            if len(_files) == 0:
                raise ValueError("Cannot find the dinov3 weights. Please read dinov3easy/checkpoints/download_instructions.md.")
        else:
            raise ValueError("Cannot find the dinov3 weights. Please read dinov3easy/checkpoints/download_instructions.md.")
    else:
        raise ValueError("Location specified in dinov3easy/checkpoints/checkpoints_location.txt not found. Please read dinov3easy/checkpoints/download_instructions.md.")



# MODEL_CHECKPOINTS_PATHS_DICT

_pth_files = [f for f in os.listdir(MODEL_CHECKPOINTS_PATH) if f.endswith(".pth")]
_pth_sat_files = [f for f in _pth_files if "sat" in f]

def _get_corresponding_file(model_name: str):
    _lookin_list = []
    if "sat" in model_name:
        _lookin_list = _pth_sat_files
    else:
        _lookin_list = _pth_files

    for f in _lookin_list:
        if model_name in f:
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
    MODEL_CHECKPOINTS_PATHS_DICT[_key] = _get_corresponding_file(_key)
    if MODEL_CHECKPOINTS_PATHS_DICT[_key] is None:
        raise ValueError(f"Cannot find the checkpoint for model {_key}. Please investigate and read dinov3easy/checkpoints/download_instructions.md.")
