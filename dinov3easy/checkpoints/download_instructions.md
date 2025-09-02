# How to get model weights

## I already have them in a local directory

If you already have the model weights downloaded, you can specify the full path to the directory containing the weights in the `checkpoint_location.txt` file, or simply running the `python ./dinov3easy/setup/checkpoints.py` script after you have activated the appropriate python environment (but should work with any python >= 3.10).

If you installed this repo as a package (following the instructions in the main readme), then you can create an empty python file and copy-paste and run the following code:

```py
# If you have a screen on your machine
import dinov3easy
dinov3easy.setup.checkpoints.run()
```

```py
# On a headless server
import dinov3easy
dinov3easy.setup.checkpoints.run(use_gui=False)
```

Follow the instructions.

## I have to download them

Be sure to have at least 65GB of space on the disk where you installed this repo. You can also just download some of them, it is not necessary to have them all.

Go to the [original dinov3 repository](https://github.com/facebookresearch/dinov3) and follow the instructions there to download the model weights. You can put them anywhere you want. After downloading, follow the instructions of the section above.

## On a server

If you're working on a headless server (without screen or GUI), simply run:

`python ./dinov3easy/setup/checkpoints.py --no-gui`