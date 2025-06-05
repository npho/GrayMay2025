# Gray May 2025

**Nels Schimek** and **Nam Pho**

May is Brain Tumor Awareness Month, and this project aims to raise awareness about brain tumors through the use of machine learning (ML) and artificial intelligence (AI) techniques.

Lecture content from *Neural Network Methods for Signals in Engineering and Physical Sciences* (PHYS 417, Spring Quarter 2025) at the University of Washington can provide a foundation for the methods used in this project. Alternatively, any PyTorch and Python tutorials can be used as well.

### Data Set

A MRI scan data set with brain images of various brain tumors (i.e., glioma, meningioma, pituitary) and normal scans from Kaggle [[www](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)].

```
data (7,022 images)
├── Training (5,712 images)
│   ├── notumor (1,595 images)
│   │   ├── Te-no_0000.jpg
│   │   └── ...
│   ├── glioma (1,321 images)
│   │   ├── Te-gl_0000.jpg
│   │   └── ...
│   ├── meningioma (1,339 images)
│   │   ├── Te-me_0000.jpg
│   │   └── ...
│   └── pituitary (1,457 images)
│       ├── Te-pi_0000.jpg
│       └── ...
└── Testing (1,311 images)
    ├── notumor (405 images)
    │   ├── Te-no_0000.jpg
    │   └── ...
    ├── glioma (300 images)
    │   ├── Te-gl_0000.jpg
    │   └── ...
    ├── meningioma (306 images)
    │   ├── Te-me_0000.jpg
    │   └── ...
    └── pituitary (300 images)
        ├── Te-pi_0000.jpg
        └── ...
```

The `KaggleBrainDataset.py` script will load and prepare the data set. Please provide the path for downloading and loading images. It will default to a `data` folder in the current working directory through the `KAGGLEHUB_CACHE` environment variable.

### Methods

Exploring the use of convolutional neural networks (CNNs) for image classification as well as Vision Transformers (ViTs) for the same task.

```
$ python train.py --help
usage: train.py [-h] [-m {BrainTumorNet,ViT}] [-d {cpu,cuda,mps}] [-e EPOCHS] [--split SPLIT]
                [-b BATCH_SIZE] [-l LEARNING_RATE] [--decay DECAY] [-w WEIGHTS] [-v]
                [-p PRETRAIN]

Kaggle Brain Tumor MRI Dataset Training Script

optional arguments:
  -h, --help            show this help message and exit
  -m {BrainTumorNet,ViT}, --model {BrainTumorNet,ViT}
                        Model selection.
  -d {cpu,cuda,mps}, --dev {cpu,cuda,mps}, --device {cpu,cuda,mps}
                        Device to use for training [cpu, gpu, mps], defaults to automatic
                        detection.
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs to train the model.
  --split SPLIT         Train-test split ratio, defaults to 0.8 (80% train, 20% validation).
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size for training.
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate for the optimizer.
  --decay DECAY         Weight decay for the optimizer.
  -w WEIGHTS, --weights WEIGHTS
                        Path to store or load the model weights file, if any.
  -v, --verbose
  -p PRETRAIN, --pretrain PRETRAIN
                        Path to pre-trained model, if any, For ViT.
$ 
```

### Results

![BrainTumorNet Training Plot](/fig/training-BrainTumorNet.png)

![BrainTumorNet Validation Plot](/fig/validation-BrainTumorNet.png)

![BrainTumorNet Confusion Matrix](/fig/confusion-BrainTumorNet.png)
