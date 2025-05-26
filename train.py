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

def train(model, epochs, data, device, loss_func, optimizer):
	loss_epoch = []
	validation_epoch = []

	# Stage model on whatever device we are using
	model.to(device)

	# Prettier tqdm progress bar
	pbar_epoch = tqdm.tqdm(iterable=range(epochs), colour="green", desc="Epoch")
	for epoch in pbar_epoch:
		loss_batch = []
		validation_batch = []

		# batches (data.dataset.length)
		pbar_batch = tqdm.tqdm(total=len(data), colour="blue", desc="Batch", leave=False)
		for batch, samples in enumerate(data):
			images, labels = samples
			images.to(device)
			labels.to(device)

			train_outputs = model(images)
			loss = loss_func(train_outputs, labels)
			loss_batch.append(loss.item())
			
			# Batch-level backpropagation
			optimizer.zero_grad() # Zero gradients from previous epoch
			loss.backward() # Calculate gradient
			optimizer.step() # Update weights

			"""
			# Compute Validation Accuracy
			with torch.no_grad():
				validation_outputs = model(validation_features)
				_, predicted = torch.max(validation_outputs, 1)
			"""
			validation_batch.append(0)

			pbar_batch.set_postfix({
					"Loss" : loss_batch[batch],
					"Acc" : validation_batch[batch],
				})
			pbar_batch.update(1)
		
		pbar_batch.close()

		# Save batch stats at the epoch level
		loss_epoch.append(loss_batch)
		validation_epoch.append(validation_batch)

		# Display loss and accuracy in tqdm progress bar
		pbar_epoch.set_postfix({
				"Loss" : np.mean(loss_epoch[epoch]), 
				"Acc" : np.mean(validation_epoch[epoch])
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
	parser.add_argument("-l", "--learning_rate",
					default=0.001, 
					type=float, 
					help="Learning rate for the optimizer.")
	parser.add_argument("--decay",
					default=0.0001, 
					type=float, 
					help="Weight decay for the optimizer.")
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

	# https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
	data_train = DataLoader(
						kaggle, 
						batch_size=args.batch_size, 
						shuffle=True,
						num_workers=os.cpu_count()
					)
	
	# TODO figure out training/validation

	# Initialize model
	model = None
	
	# Instantiate model choice
	if args.model == "ViT":
		# Nels TODO initiatiate model
		pass
		
		# Nels TODO train model
		pass 
	else:
		model = BrainTumorNet() # Default
		print(model)

		# Hyperparameters
		loss_func = torch.nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay)

		### Training and other stuff
		train(	
			model=model, 
			epochs=args.epochs,
			data=data_train, 
			device=device,
			loss_func=loss_func,
			optimizer=optimizer
		)
