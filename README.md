# DINOv3easy: Your DINOv3 Companion!

Simple, self-contained wrapper around Meta's DINO v3 repo that simlifies model loading and features analysis.

Some considerations that might prove useful:

- 



## How to use it

This is an installable package, so just do:

```bash
# create the venv of your choice and activate it
python -m pip install git+https://github.com/Matteoleccardi/DINOv3easy.git
```

That's it!

Now, let's go through the package structure real quick:

```bash
DINOv3easy/
├── dino/
│   ├── __init__.py
│   ├── dino.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   └── test_dino.py
└── README.md
```

dino -> dino -> layers -> attention modified from the original one to compute the attention matrix explicitly upon request, useful for later visualization.
