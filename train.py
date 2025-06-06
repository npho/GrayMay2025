"""
train.py (May 2025)
"""

import os
import argparse
import logging
import numpy as np
import tqdm

import matplotlib.pyplot as plt

from KaggleBrainDataset import KaggleBrainDataset
from KaggleBrainDataset import ReduceChannel, EnsureRGB

from BrainTumorNet import BrainTumorNet
from TumorViT import TumorViT

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
		req_dev: [None, "cpu", "cuda", "mps"]
	"""
	device = None

	if req_dev:
		# Attempt manual device selection
		if req_dev == "mps" and torch.backends.mps.is_available():
			device = torch.device("mps")  # Apple Silicon GPU
		elif req_dev == "cuda" and torch.cuda.is_available():
			device = torch.device("cuda")  # NVIDIA GPU
		else:
			device = torch.device("cpu")  # Defaults to CPU
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
	train_loss_epoch = []
	val_loss_epoch = []
	val_accuracy_epoch = []

	# Stage model on whatever device we are using
	model.to(device)

	# Split out the training and validation DataLoaders
	data_train, data_val = data

	# Prettier tqdm progress bar
	pbar_epoch = tqdm.tqdm(iterable=range(epochs), colour="green", desc="Epoch")
	for epoch in pbar_epoch:
		train_loss_batch = []
		val_loss_batch = []
		val_accuracy_batch = []

		# batches (data.dataset.length)
		pbar_batch = tqdm.tqdm(total=len(data_train), colour="blue", desc="Batch", leave=False)
		for _, (images, labels) in enumerate(data_train):
			images = images.to(device)
			labels = labels.to(device)

			train_outputs = model(images)
			loss = loss_func(train_outputs, labels)
			train_loss_batch.append(loss.item()) # batch loss
			
			# Batch-level backpropagation
			optimizer.zero_grad() # Zero gradients from previous epoch
			loss.backward() # Calculate gradient
			optimizer.step() # Update weights

			#if batch % 35 == 0 and batch != 0:
			# Compute validation accuracy
			val_pred = []
			val_lbls = []
			val_loss = []
			with torch.no_grad():
				for _, (images, labels) in enumerate(data_val):
					images = images.to(device)
					labels = labels.to(device)

					val_outputs = model(images)
					val_loss_iter = loss_func(val_outputs, labels)
					val_loss.append(val_loss_iter.item())
					val_pred += model(images).cpu().tolist()
					val_lbls += labels.cpu().tolist()
			
			# numpy object for vectorization
			val_pred = np.array(val_pred).argmax(axis=1)
			val_lbls = np.array(val_lbls)

			# Get validation accuracy
			val_accuracy_batch.append((val_pred == val_lbls).mean())
			val_loss_batch.append(np.array(val_loss).mean())

			pbar_batch.set_postfix({
					"Loss": train_loss_batch[-1],
					"Acc": val_accuracy_batch[-1],
				})
				
			pbar_batch.update(1)

			# Save weights at the end of each batch
			if weights:
				# logging.info(f"Checkpoint model to {weights}")
				torch.save(model.state_dict(), weights)
		
		pbar_batch.close()

		# Save batch stats at the epoch level
		val_loss_epoch.append(np.mean(val_loss_batch))
		val_accuracy_epoch.append(np.mean(val_accuracy_batch))
		train_loss_epoch.append(np.mean(train_loss_batch))

		# Display loss and accuracy in tqdm progress bar
		pbar_epoch.set_postfix({
				"Loss": np.mean(val_loss_epoch[epoch]), 
				"Acc": np.mean(val_accuracy_epoch[epoch])
			})
		
		# Plot training curves
		epochs = range(1, len(val_accuracy_epoch)+1)
		plt.figure(figsize=(8, 5))
		plt.plot(epochs, val_accuracy_epoch, marker='o', color='blue', label='Validation Accuracy')
		plt.title('Validation Accuracy per Epoch')
		plt.xlabel('Epoch')
		plt.ylabel('Validation Accuracy (%)')
		plt.grid(True)
		plt.legend()
		plt.savefig('fig/validation.png')
		plt.close()

		plt.figure(figsize=(8, 5))
		plt.plot(epochs, val_loss_epoch, marker='o', color='blue', label='Validation Loss')
		plt.plot(epochs, train_loss_epoch, marker='o', color='red', label='Training Loss')
		plt.title('Loss per Epoch')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.grid(True)
		plt.legend()
		plt.savefig('fig/training.png')
		plt.close()

	return val_accuracy_epoch, val_loss_epoch, train_loss_epoch
		

def generate_dataloaders(train=True, transforms=None, num_workers=4):

	# Load the dataset
	kaggle = KaggleBrainDataset(train=train, transform=transforms)

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
						num_workers=min(num_workers, os.cpu_count())
					)

	# Creates the DataLoader for the validation split
	data_val = DataLoader(
						data_val,
						batch_size=args.batch_size,
						shuffle=True,
						num_workers=min(num_workers, os.cpu_count())
					)
	
	return data_train, data_val


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
					choices=["cpu", "cuda", "mps"],
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
	parser.add_argument("-p", "--pretrain",
					 	default=None,
						type=str,
						help="Path to pre-trained model, if any, For ViT.")

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
	if args.model == "BrainTumorNet":
		transforms = v2.Compose([
			ReduceChannel(),
			v2.ToDtype(torch.float32, scale=True),
			v2.Normalize(mean=[0], std=[1]),
			v2.Resize(size=(512, 512)),
			v2.RandomHorizontalFlip(p=0.5),
		])
	else:
		transforms = v2.Compose([
			EnsureRGB(),
			v2.ToDtype(torch.float32, scale=True),
			v2.Resize(size=(224, 224)),
			v2.RandomHorizontalFlip(p=0.5),
			v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # ImageNet-style
			])
		
	# Make data loaders
	data_train, data_val = generate_dataloaders(train=True, transforms=transforms)

	# Initialize model
	model = None
	
	# Instantiate model choice
	if args.model == "ViT":
		if args.pretrain:
			model = TumorViT(vit_path=args.pretrain)
		else:
			# default model from Hugging Face
			model = TumorViT(vit_path="google/vit-large-patch16-224")
	else:
		data_train, data_val = generate_dataloaders(train=True, transforms=transforms)
		model = BrainTumorNet()  # Default
	
	logging.info(f"Testing the {args.model} model:\n{model}")

	# Determine model size
	# https://www.geeksforgeeks.org/check-the-total-number-of-parameters-in-a-pytorch-model/
	total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	logging.info(f'Model total parameters: {total_params}')
	total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	logging.info(f'Model trainable parameters: {total_params}')

	# Resume checkpoint training
	if args.weights:
		if os.path.exists(args.weights):
			logger.info(f"Using model weights: {args.weights}")
			weights = torch.load(args.weights, weights_only=True)
			model.load_state_dict(weights)
		else:
			logging.info(f"Does {args.weights} exist? Training from scratch.")

	# Hyperparameters
	loss_func = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay)

	# Training and other stuff
	val_accuracy, val_loss, train_loss = train(
		model=model,
		weights=args.weights,
		epochs=args.epochs,
		data=[data_train, data_val],
		device=device,
		loss_func=loss_func,
		optimizer=optimizer
	)
