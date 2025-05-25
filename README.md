# Gray May 2025

**Nels Schimek** and **Nam Pho**

May is Brain Tumor Awareness Month, and this project aims to raise awareness about brain tumors through the use of machine learning (ML) and artificial intelligence (AI) techniques.

Lecture content from *Neural Network Methods for Signals in Engineering and Physical Sciences* (PHYS 417, Spring Quarter 2025) at the University of Washington can provide a foundation for the methods used in this project. Alternatively, any PyTorch and Python tutorials can be used as well.

### Data Set

A neuroscience data set with brain images of tumor-normal pairs.

To prepare the data set, unzip it in the current directory. The `.gitignore` file should prevent the `tumor_diagnosis` directory from being tracked by Git.

```bash
unzip data-tumor.zip
```

The resulting directory is __ images.

```
tumor_diagnosis
└── test
│   ├── 0_0.png
│   ├── ...
│   └── 0_1.png
└── train_normal
│   ├── 0_0.png
│   ├── ...
│   └── 0_1.png
└── train_tumorous
    ├── 0_0.png
    ├── ...
    └── 0_1.png
```

### Methods

Exploring the use of convolutional neural networks (CNNs) for image classification as well as Vision Transformers (ViTs) for the same task.

```bash
python train.py --help
```


