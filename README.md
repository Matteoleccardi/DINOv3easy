<p align="center">
  <img src="./assets/easy_dino.png" alt="A relaxed dinosaur">
</p>

# dinov3easy: Your DINOv3 Companion!

Simple, self-contained wrapper around Meta's DINO v3 repo that simplifies model loading and features analysis.

This package also includes the [original dinov3 repository](https://github.com/facebookresearch/dinov3) from facebookresearch. The repo was slighly modified to add easier visualization and inspection functionality (see the *Note that...* section down below).

## Get Started with DINO v3

This is an installable package, so just do:

```bash
# create the venv of your choice and activate it, then run
python -m pip install git+https://github.com/Matteoleccardi/DINOv3easy.git
```

Note that if you want to use this package with GPU support for pytorch, now you have to -reinstall pytorch on your own (see [pytorch website](https://pytorch.org/get-started/locally/)):

```bash
# Choose the cuda version according to your gpu drivers (run nvidia-smi to find out the closest version)
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

*Note*: If you encounter error during installation, it is probably an out-of-space error. Run this command below:

```bash
TMPDIR=/path/to/dir/with/lots/of/free/space python -m pip install git+https://github.com/Matteoleccardi/DINOv3easy.git
# after installation, you can delete that dir, but it will probably be empty
```

## Get your weights

You will have to download the dinov3 weights following the instructions in the [original dinov3 repository](https://github.com/facebookresearch/dinov3). By default here, only the weights of `dinov3_vits16` are available (~80 MB), the others are too heavy for github to store.

After downloading, please follow the quick steps explained in [dinov3easy/checkpoints/download_instructions](./dinov3easy/checkpoints/download_instructions.md)

## How to use it

You can install the repo and use the helper functions found in the many sub-modules.

For quick, interactive visualization, you can run the scripts found in `./dinov3easy/view/interactive*` directly, or you can import them in another script and run the `run()` method.

See the [examples](./examples/EXAMPLES.md) to get an idea on what you can do with this, or check the source code of the scripts in `dinov3easy/view/`.

## Note that...

`dinov3` -> `dinov3` -> `layers` -> `attention` modified from the original one to compute the attention matrix explicitly upon request, useful for later visualization.

## Contact

If you find bugs or something does not work, open an Issue on github at [this link](https://github.com/Matteoleccardi/DINOv3easy), or contact me at `matteo.leccardi@polimi.it`.

Enjoy!
*By Matteo Leccardi and Nico Schulthess.*