"""
test.py (May 2025)
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

def test(model, data, device):
	"""
	Test the model on the provided dataset.
	Args:
		model: The neural network model to be tested.
		data: DataLoader containing the test dataset.
		device: The device (CPU/GPU) to run the model on.
	"""
	# Prepare the model
	model.eval() # Inference mode
	model.to(device) # Model to device

	# Compute validation accuracy
	test_pred = []
	test_lbls = []
	
	pbar = tqdm.tqdm(total=len(data), colour="blue", desc="Image")
	with torch.no_grad():
		for _, (images, labels) in enumerate(data):
			images = images.to(device)
			labels = labels.to(device)

			test_pred += model(images).cpu().tolist()
			test_lbls += labels.cpu().tolist()

			pbar.update(1)

	pbar.close()
	
	# numpy object for vectorization
	test_pred = np.array(test_pred).argmax(axis=1)
	test_lbls = np.array(test_lbls)

	# Final result!
	accuracy = (test_pred == test_lbls).mean()
	print("[+] Accuracy: {:.2f}%".format(accuracy * 100))

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
					help="Device to use for testing [cpu, gpu, mps], defaults to automatic detection.")
	parser.add_argument("-b", "--batch_size",
					default=1, 
					type=int, 
					help="Batch size for testing.")				
	parser.add_argument('-v', '--verbose', action='store_true')

	args = parser.parse_args()

	# Set up logging
	logger = logging.getLogger()
	logger.setLevel(logging.NOTSET if args.verbose else logging.WARNING)

	### Figure out the device to use for training
	device = device_check(args.dev)
	logger.info(f"[+] Using device: {device}")

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

	# Do model evaluation
	logging.info(f"[+] Your model is {args.model}:")
	logging.info(model)

	# Re-use transformations from training
	transforms = v2.Compose([
		ReduceChannel(),
		v2.ToDtype(torch.float32, scale=True),
		v2.Normalize(mean=[0], std=[1]),
		v2.Resize(size=(512, 512)),
		v2.RandomHorizontalFlip(p=0.5),
	])

	# Load the testing dataset
	kaggle = KaggleBrainDataset(train=False, transform=transforms)

	# Load testing data via DataLoader
	data_test = DataLoader(
						kaggle, 
						batch_size=args.batch_size, 
						shuffle=True,
						num_workers=os.cpu_count()
					)

	### Testing
	test(	
		model=model, 
		data=data_test,
		device=device
	)
