"""
KaggleBrainDataset.py (May 2025)
"""

import os
import kagglehub

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2

# https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_custom_transforms.html


class ReduceChannel(torch.nn.Module):
	def forward(self, img, c=1):
		_, h, w = img.shape
		img = img[0].reshape((c, h, w))
		return img
	

class EnsureRGB(torch.nn.Module):
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[0] == 1:
            # Grayscale → RGB by repeating
            return img.repeat(3, 1, 1)
        elif img.shape[0] == 4:
            # RGBA → RGB by slicing off alpha
            return img[:3, :, :]
        elif img.shape[0] == 3:
            # Already RGB
            return img
        else:
            raise ValueError(f"Unsupported number of channels: {img.shape[0]}")

# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html


class KaggleBrainDataset(Dataset):
	"""The Kaggle Brain Tumor MRI Dataset."""

	def __init__(self, path="./data", train=True, transform=None, force=False):
		self.path = path
		self.train = train
		self.transform = transform

		self.label_map = [
				"notumor", 
				"glioma", 
				"meningioma", 
				"pituitary"
			]

		# Download the data with kagglehub
		# https://github.com/Kaggle/kagglehub
		os.environ['KAGGLEHUB_CACHE'] = self.path
		self.path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset", force_download=force)

		# Specify folder for training or testing
		folder = "Training" if self.train else "Testing"
		self.path = os.path.join(self.path, folder)

		# Count up all the image files
		self.file = []
		self.label_txt = []
		self.label_idx = []
		
		for root, dirs, files in os.walk(self.path):
			if len(files) > 0:
				# Save up image file paths
				self.file += [os.path.join(root, file) for file in files]
				
				# Derive image labels from directory name
				label_txt = os.path.basename(root) # text label
				label_idx = self.label_map.index(label_txt) # numeric label

				# Store both text and numeric labels
				self.label_txt += [label_txt] * len(files)
				self.label_idx += [label_idx] * len(files)
		
		assert len(self.file) == len(self.label_idx), "Image and label count mismatch!!!"

		self.length = len(self.label_idx)

	def __len__(self):		
		return self.length
	
	def __getitem__(self, idx):
		"""Get an image and its label by index."""

		# Get image and apply transformations
		image = read_image(self.file[idx])
		if self.transform:
			image = self.transform(image)

		# Get image label and return
		label = self.label_idx[idx]

		return image, label

if __name__ == "__main__":
	print("The Kaggle Brain Tumor Dataset")

	# https://docs.pytorch.org/vision/master/transforms.html#v2-api-reference-recommended
	transforms = v2.Compose([
		ReduceChannel(),
		v2.ToDtype(torch.float32, scale=True),
		v2.Normalize(mean=[0], std=[1]),
		v2.Resize(size=(512, 512)),
		v2.RandomHorizontalFlip(p=0.5),
	])

	# Demonstrate training data set
	kaggle = KaggleBrainDataset(transform=transforms)
	print(f"\tNumber of training images: {len(kaggle)}")

	# Demonstrate testing data set
	kaggle = KaggleBrainDataset(train=False, transform=transforms)
	print(f"\tNumber of testing images: {len(kaggle)}")
