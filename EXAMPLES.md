# Some usage examples to start with

dinov3easy is a wrapper around the dinov3 repository. If you want to inspect the original repository, just open the dinov3easy/dinov3 folder.

This package makes it somewhat easier to work with the pretrained dinov3 models, and visualize things interactively.

Before attempting any of this, be sure to read and follow the instructions in the [model checkpoints download and setup instructions](./dinov3easy/checkpoints/download_instructions.md)

## Loading a dinov3 model

```py
from dinov3easy.load import load_dinov3_models
model = load_dinov3_models(model_type)
```

Simple as that!

`model_type` can be chosen from the following list:

- ViT models (can have attention map)
    - `'dinov3_vits16'`
    - `'dinov3_vits16plus'`
    - `'dinov3_vitb16'`
    - `'dinov3_vitl16'`
    - `'dinov3_vith16plus'`
    - `'dinov3_vit7b16'`
- ConvNeXt models (no attention map)
    - `'dinov3_convnext_tiny'`
    - `'dinov3_convnext_small'`
    - `'dinov3_convnext_base'`
    - `'dinov3_convnext_large'`
- ViT for satellite images
    - `'dinov3_vitl16_sat'`
    - `'dinov3_vit7b16_sat'`


## Inference with dinov3

Inference can be easy. Just use the following helper fucntions.

```py
from dinov3easy.utils.constants import IMAGENET_MEAN, IMAGENET_STD
from dinov3easy.utils.predict import get_features, get_attention_map

# Let's say you opened an image somehow, and transformed it into a
# torch.Tensor x
# (See scripts in the dinov3easy.view folder to see some complete pipelines)
image_side = 512
x = torch.randint(0, 255, size=(1, 3, image_side, image_side))

# The image given to dinov3 models must be a 2D, three channels (RGB) image
# image_side must be muliple of 16 for ViT-based models, while multiple of 32 for ConvNeXt-based models.

# --- IMPORTANT ---
# Image should first be normalized (between 0 and 1) and then
# standardized before passing it for inference:
x = x.float() # whatever it is, get it to be float
x = x / 255 # whatever it is, get it in the [0;1] range
m = torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)
s = torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
x = (x - m) / s # whatever it is, standardize it with these values

# -> to get only the features (1, C, image_side/16 or 32, image_side/16 or 32)
features = get_features(model, x)

# -> to get both features and dino output with shape (1, C)
features, out = get_features(model, x, dino_output=True)
```

## Interactive viewers

You can run view scripts from the command line, or from within a python file.

To run them from within a python file (easyer if you installed the package), do the following:

**Attention**

Opens an image (better if square or almost square) and let's the user click on the image to select a patch. The overlay will show to which patches the selected patch is considering the most. Works only with ViT-based models.

```py
from dinov3easy.view.interactive_attention import run
run()
```

**PCA of features**

Opens an image (better if square or almost square) and shows the Principal Components Analysis (PCA) decomposition of the features extracted from the image.

```py
from dinov3easy.view.interactive_pca import run
run()
```

**Cosine similarity of features**

Opens an image (better if square or almost square) and lets the user select patches in the image. Colors indicate to which other patches the selected patch is most similar to (left continuous, right discretization into fixed intervals of similarity).

```py
from dinov3easy.view.interactive_similarity import run
run()
```

**Predicted class similarity**

Opens an image (better if square or almost square) and shows to which patches the output of the model (the one with shape `(1, C)`) is most similar to.
In my understanding, this is quite equivalent to asking the model which patches are the most influential in understanding the image as a whole.

```py
from dinov3easy.view.interactive_class_similarity import run
run()
```

**Note that:**

You can also specify the dinov3 model, image resize and image path directly to the run() function instead of using the interactive GUI (use full path): 

```py
# Just an example, it holds for all
run(model_type='dinov3_vitb16', img_resize=1024, img_path = '/full/path/to/image.jpeg')
```

## How to deal with medical images?

You have to find a way to transform your image intensities, whatever they are, into a \[0;255\] or \[0;1\] range that makes sense.
You can also use specific mean and standard deviation to standardize the input, but results can vary since dino was trained using specifically those values. 