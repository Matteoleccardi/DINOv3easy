# How to get model weights

## I already have them in a local directory

If you already have the model weights downloaded, you can specify the full path to the directory containing the weights in the `checkpoint_location.txt` file.

## I have to download them

Be sure to have at least 65GB of space on the disk where you installed this repo.
After that, run the following python script:

```bash

python -m dinov3easy.setup.download_checkpoints
```