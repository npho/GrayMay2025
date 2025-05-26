"""
train.py (May 2025)
"""

import os
import argparse
import logging
import numpy as np
import tqdm

from KaggleBrainDataset import KaggleBrainDataset
from KaggleBrainDataset import ReduceChannel

from BrainTumorNet import BrainTumorNet

import torch
from torch.utils.data import random_split
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

def train(model, weights, epochs, data, device, loss_func, optimizer):
	loss_epoch = []
	validation_epoch = []

	# Stage model on whatever device we are using
	model.to(device)

	# Split out the training and validation DataLoaders
	data_train, data_val = data

	# Prettier tqdm progress bar
	pbar_epoch = tqdm.tqdm(iterable=range(epochs), colour="green", desc="Epoch")
	for epoch in pbar_epoch:
		loss_batch = []
		validation_batch = []

		# batches (data.dataset.length)
		pbar_batch = tqdm.tqdm(total=len(data_train), colour="blue", desc="Batch", leave=False)
		for batch, (images, labels) in enumerate(data_train):
			images = images.to(device)
			labels = labels.to(device)

			train_outputs = model(images)
			loss = loss_func(train_outputs, labels)
			loss_batch.append(loss.item()) # batch loss
			
			# Batch-level backpropagation
			optimizer.zero_grad() # Zero gradients from previous epoch
			loss.backward() # Calculate gradient
			optimizer.step() # Update weights

			# Compute validation accuracy
			val_pred = []
			val_lbls = []
			with torch.no_grad():
				for _, (images, labels) in enumerate(data_val):
					images = images.to(device)
					labels = labels.to(device)
				
					val_pred += model(images).cpu().tolist()
					val_lbls += labels.cpu().tolist()
			
			# numpy object for vectorization
			val_pred = np.array(val_pred).argmax(axis=1)
			val_lbls = np.array(val_lbls)

			# Get validation accuracy
			validation_batch.append((val_pred == val_lbls).mean())

			pbar_batch.set_postfix({
					"Loss" : loss_batch[batch],
					"Acc" : validation_batch[batch],
				})
			pbar_batch.update(1)

			# Save weights at the end of each batch
			if weights:
				logging.info(f"Checkpoint model to {weights}")
				torch.save(model.state_dict(), weights)
		
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
	parser.add_argument("--split",
					default=0.8,
					type=float,
					help="Train-test split ratio, defaults to 0.8 (80%% train, 20%% validation).")
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
	parser.add_argument("-w", "--weights",
					default=None, 
					type=str, 
					help="Path to store or load the model weights file, if any.")
	parser.add_argument('-v', '--verbose', action='store_true')

	args = parser.parse_args()

	# Set up logging
	logging.basicConfig(format="[+] %(message)s")
	logger = logging.getLogger()
	logger.setLevel(logging.NOTSET if args.verbose else logging.WARNING)

	### Figure out the device to use for training
	device = device_check(args.dev)
	logger.info(f"Using device: {device}")

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

	# train-test split
	n_train = int(len(kaggle) * args.split)
	n_val = len(kaggle) - n_train
	data_train, data_val = random_split(kaggle, [n_train, n_val])

	# https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
	
	# Creates the DataLoader for the training split
	data_train = DataLoader(
						data_train, 
						batch_size=args.batch_size, 
						shuffle=True,
						num_workers=os.cpu_count()
					)

	# Creates the DataLoader for the validation split					
	data_val = DataLoader(
						data_val, 
						batch_size=args.batch_size, 
						shuffle=True,
						num_workers=os.cpu_count()
					)

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
		logging.info(f"Testing the {args.model} model:\n{model}")

		# Resume checkpoint training
		if os.path.exists(args.weights):
			logger.info(f"Using model weights: {args.weights}")
			weights = torch.load(args.weights, weights_only=True)
			model.load_state_dict(weights)

		# Hyperparameters
		loss_func = torch.nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay)

		### Training and other stuff
		train(	
			model=model,
			weights=args.weights,
			epochs=args.epochs,
			data=[data_train, data_val], 
			device=device,
			loss_func=loss_func,
			optimizer=optimizer
		)
