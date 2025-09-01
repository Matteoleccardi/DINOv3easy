# How to get model weights

## I already have them in a local directory

If you already have the model weights downloaded, you can specify the full path to the directory containing the weights in the `checkpoint_location.txt` file, or simply running the `python ./dinov3easy/setup/checkpoints.py` script after you have activated the appropriate python environment (but should work with any python >= 3.10).

## I have to download them

Be sure to have at least 65GB of space on the disk where you installed this repo.

Go to the [original dinov3 repository](https://github.com/facebookresearch/dinov3) and follow the instructions there to download the model weights. You can put them anywhere you want, however the preferred directory would be this one. After downloading, launch the interactive `python ./dinov3easy/setup/checkpoints.py` script to configure the checkpoints.

## On a server

If you're working on a headless server (without screen or GUI), simply run:

`python ./dinov3easy/setup/checkpoints.py --no-gui`