"""
train.py (May 2025)
"""

import os
import argparse
import numpy as np
import tqdm

from KaggleBrainDataset import KaggleBrainDataset
from KaggleBrainDataset import ReduceChannel

from BrainTumorNet import BrainTumorNet

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

# numpy and pytorch seeds for reproducibility 
np.random.seed(0)
torch.manual_seed(0)

def device_check(req_dev):
	"""
	Decides the device being used for training.
	Returns:
		req_dev: [None, "cpu", "gpu", "mps"]
	"""
	device = None

	if req_dev:
		# Attempt manual device selection
		if req_dev == "mps" and torch.backends.mps.is_available():
			device = torch.device("mps") # Apple Silicon GPU
		elif req_dev == "cuda" and torch.cuda.is_available():
			device = torch.device("cuda") # NVIDIA GPU
		else:
			device = torch.device("cpu") # Defaults to CPU
	else:
		# Automatic device detection 
		if torch.backends.mps.is_available():
			# Check and use Apple Silicon GPU
			# https://pytorch.org/docs/stable/notes/mps.html
			device = torch.device("mps")
		elif torch.cuda.is_available():
			# The provided code for CUDA
			device = torch.device("cuda")
		else:
			# Default to CPU if no accelerator available
			device = torch.device("cpu")

	return device

def train(model, epochs, data, device):
	train_loss_list = np.zeros(epochs)
	validation_accuracy_list = np.zeros(epochs)

	# Stage model on whatever device we are using
	model.to(device)

	# Prettier tqdm progress bar
	pbar = tqdm.tqdm(iterable=range(epochs), colour="green", desc="Epoch")

	for epoch in pbar:
		# batches
		for i, samples in enumerate(data):
			images, labels = samples
			images.to(device)
			labels.to(device)
			
			print(f"Batch {i}: Images shape: {images.shape}, Labels shape: {labels.shape}")

			if i == 1:
				break
		
		# Display loss and accuracy in tqdm progress bar
		pbar.set_postfix({
				# full-batch gradient descent version
				#"Loss" : train_loss_list[epoch], 
				#"Accuracy" : validation_accuracy_list[epoch]

				# mini-batch gradient descent version
				#"Loss" : train_loss_list[z], 
				#"Accuracy" : validation_accuracy_list[z]
			})

if __name__ == "__main__":
	msg = "Kaggle Brain Tumor MRI Dataset Training Script"
	parser = argparse.ArgumentParser(description=msg)

	# Adding arguments
	parser.add_argument("-m", "--model", 
					default="BrainTumorNet",
					choices=["BrainTumorNet", "ViT"],
					type=str,
					help="Model selection.")
	parser.add_argument("-d", "--dev", "--device", 
					default=None, 
					choices=["cpu", "gpu", "mps"],
					type=str,
					help="Device to use for training [cpu, gpu, mps], defaults to automatic detection.")
	parser.add_argument("-e", "--epochs",
					default=10, 
					type=int, 
					help="Number of epochs to train the model.")
	parser.add_argument("-b", "--batch_size",
					default=64, 
					type=int, 
					help="Batch size for training.")
	parser.add_argument('-v', '--verbose', action='store_true')

	args = parser.parse_args()

	### Figure out the device to use for training
	device = device_check(args.dev)
	print(f"[+] Using device: {device}")

	# Define image transformations
	# https://docs.pytorch.org/vision/master/transforms.html#v2-api-reference-recommended
	transforms = v2.Compose([
		ReduceChannel(),
		v2.ToDtype(torch.float32, scale=True),
		v2.Normalize(mean=[0], std=[1]),
		v2.Resize(size=(512, 512)),
		v2.RandomHorizontalFlip(p=0.5),
	])

	# Load the dataset
	kaggle = KaggleBrainDataset(train=True, transform=transforms)
	"""
	for i in range(len(kaggle)):
		image, label = kaggle[i]
		print(f"Image {i}: {image.shape}, Label: {label}")
	"""

	# https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
	data_train = DataLoader(
						kaggle, 
						batch_size=args.batch_size, 
						shuffle=True,
						num_workers=os.cpu_count()
					)
	
	# Initialize model
	model = None
	
	# Instantiate model choice
	if args.model == "BrainTumorNet":
		model = BrainTumorNet()
	elif args.model == "ViT":
		pass # Nels TODO
	else:
		model = BrainTumorNet() # Default

	print(model)
		
	### Training and other stuff
	train(	
		model=model, 
		epochs=args.epochs, 
		data=data_train, 
		device=device
	)
