# DINOv3easy: Your DINOv3 Companion!

Simple, self-contained wrapper around Meta's DINO v3 repo that simlifies model loading and features analysis.

Some considerations that might prove useful:

- 



## Get Started with DINO v3

This is an installable package, so just do:

```bash
# create the venv of your choice and activate it
python -m pip install git+https://github.com/Matteoleccardi/DINOv3easy.git
```

You will have to download the dinov3 weights from the facebookresearch repository. By default here, only the weights
of `dinov3_vits16` are downloaded, the others are too heavy for github.

Please follow the quick steps explained in [dinov3easy/checkpoints/download_instructions](./dinov3easy/checkpoints/download_instructions.md)

## How to use it

You can install the repo and use the helper functions found in the many sub-modules.

For quick, interactive visualization, you can run the scripts found in `./dinov3easy/view/interactive*`.


## Note that...

dinov3 -> dinov3 -> layers -> attention modified from the original one to compute the attention matrix explicitly upon request, useful for later visualization.
