
## Loading a dinov3 model
##########################

from dinov3easy.load import load_dinov3_models
model = load_dinov3_models(model_type='dinov3_vits16')

## Pretrained Models to choose from

from dinov3easy.load.settings import MODEL_CHECKPOINTS_PATHS_DICT, AVAILABLE_MODEL_CHECKPOINTS_PATHS_DICT

print("All models:\n", MODEL_CHECKPOINTS_PATHS_DICT)
print("Models you have downloaded and linked correctly to this package:\n", AVAILABLE_MODEL_CHECKPOINTS_PATHS_DICT)
 


## Inference with dinov3
#########################

from dinov3easy.utils.constants import IMAGENET_MEAN, IMAGENET_STD
from dinov3easy.utils.predict import get_features, get_attention_map

# Let's say you opened an image somehow.
# You need to transform it into a torch.Tensor.
# Here, I create it with random content, but the shape is the correct one.
# (See scripts in the dinov3easy.view folder to see some complete pipelines)
import torch

image_side = 512
x = torch.randint(0, 255, size=(1, 3, image_side, image_side))

# The image given to dinov3 models must be a 2D, three channels (RGB) image
# image_side must be muliple of 16 for ViT-based models, while multiple of 32 for ConvNeXt-based models.

# --- IMPORTANT ---
# Image should first be normalized (between 0 and 1) and then
# standardized before passing it for inference:
x = x.float() # whatever it is, get it to be float
x = x / 255 # whatever it is, get it in the [0;1] range
m = torch.tensor(IMAGENET_MEAN)[None, :, None, None]
s = torch.tensor(IMAGENET_STD)[None, :, None, None]
x = (x - m) / s # whatever it is, standardize it with these values

# dinov3 developers use the imagenet statistics to standardize images before giving them to the model.
# If you want to use your own statistics, of course the normalization step can be skipped; however, always perform normalization.

# -> to get only the features, of shape (1, C, image_side/(16 or 32), image_side/(16 or 32))
features = get_features(model, x)

# -> to get both features and dino output, of shape (1, C)
features, out = get_features(model, x, dino_output=True)

# -> to get only the model output, of shape (1, C)
out = model(x)




## Interactive viewers
#######################


# Attention map
from dinov3easy.view.interactive_attention import run as run_am
run_am()

# PCA of features
from dinov3easy.view.interactive_pca import run as run_pca
run_pca()
# Cosine similarity of features
from dinov3easy.view.interactive_similarity import run as run_sim
run_sim()

# Predicted class similarity
from dinov3easy.view.interactive_class_similarity import run as run_cs
run_cs()



# You can also specify the dinov3 model, image resize and image path directly to the run() function 
# instead of using the interactive GUI (use full path).
# you can also run all of them together for the same image

from dinov3easy.view.all import run as run_all
run_all(model_type='dinov3_vits16', img_resize=2048, img_path = None) # img_path = '/full/path/to/image.png' or any other image format supported by pillow


# And for medical images?
# -> You can take a screenshot, or import your image and standardize it, and apply dino directly.
# Go into the code and explore the scripts in the dinov3easy.view subpackage. This is a good point where to start.

# NOTE:
# Attention map visualization is possible because i modified the original repo
# so that now there is the option of asking ther model to store the attention map of any attention layer
# to retrieve it after the forward pass.

# Enjoy!
# By Matteo Leccardi and Nico Schulthess