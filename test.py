"""
test.py (May 2025)
"""

import os
import argparse
import logging
import numpy as np
import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from KaggleBrainDataset import KaggleBrainDataset
from KaggleBrainDataset import ReduceChannel, EnsureRGB

from BrainTumorNet import BrainTumorNet
from TumorViT import TumorViT

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
		for batch, (images, labels) in enumerate(data):
			images = images.to(device)
			labels = labels.to(device)

			test_pred += model(images).cpu().tolist()
			test_lbls += labels.cpu().tolist()

			# Print figure
			# Apply softmax to get probabilities
			p = torch.nn.functional.softmax(torch.tensor(test_pred[-1]).float(), dim=0).numpy()
			
			# Plot the image and probabilities
			fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
			
			# Show the image
			images = images.reshape([512, 512]).cpu().numpy()
			ax1.imshow(images, cmap="gray")
			ax1.set_title(f"Diagnosis: {data.dataset.label_map[labels]}")
			ax1.axis('off')
			
			# Show the probabilities
			c = range(len(data.dataset.label_map))
			ax2.bar(c, p)
			ax2.set_xticks(c)
			ax2.set_xticklabels(data.dataset.label_map, rotation=45)
			ax2.set_ylim(0, 1)
			ax2.set_xlabel('Class')
			ax2.set_ylabel('Probability')
			ax2.set_title('Model Predictions')
			
			fig.tight_layout()

			fig.savefig(f"fig/img-{batch:04d}.png")  # Save to file
			plt.close()

			# Update progress bar
			pbar.update(1)

	pbar.close()
	
	# numpy object for vectorization
	test_pred = np.array(test_pred).argmax(axis=1)
	test_lbls = np.array(test_lbls)

	# Final result!
	accuracy = (test_pred == test_lbls).mean()
	print("[+] Accuracy: {:.2f}%".format(accuracy * 100))

	return test_pred, test_lbls

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
					help="Device to use for testing [cpu, gpu, mps], defaults to automatic detection.")
	parser.add_argument("-b", "--batch_size",
					default=1, 
					type=int, 
					help="Batch size for testing.")
	parser.add_argument("-w", "--weights",
					default=None, 
					type=str, 
					help="Path to the model weights file.")
	parser.add_argument('-v', '--verbose', action='store_true')

	args = parser.parse_args()

	# Set up logging
	logging.basicConfig(format="[+] %(message)s")
	logger = logging.getLogger()
	logger.setLevel(logging.NOTSET if args.verbose else logging.WARNING)

	### Figure out the device to use for training
	device = device_check(args.dev)
	logger.info(f"Using device: {device}")

	# Initialize model
	model = None
	
	# Instantiate model choice
	if args.model == "ViT":
		model = TumorViT(vit_path="google/vit-large-patch16-224")
	else:
		model = BrainTumorNet() # Default

	# Do model evaluation
	logging.info(f"Training the {args.model} model:\n{model}")

	# Re-use transformations from training
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

	# Load the testing dataset
	kaggle = KaggleBrainDataset(train=False, transform=transforms)

	# Load testing data via DataLoader
	data_test = DataLoader(
						kaggle, 
						batch_size=args.batch_size, 
						shuffle=True,
						num_workers=min(8, os.cpu_count())
					)

	# Load model weights if provided
	if args.weights:
		if os.path.exists(args.weights):
			logger.info(f"Loading model weights: {args.weights}")
			weights = torch.load(args.weights, weights_only=True)
			model.load_state_dict(weights)
		else:
			logger.warning(f"Does weights file {args.weights} exist?")
	else:
		logger.info("No weights file provided, using untrained model.")
	
	### Testing
	test_pred, test_lbls = test(	
		model=model, 
		data=data_test,
		device=device
	)

	cm = confusion_matrix(test_pred, test_lbls, labels=None)
	disp = ConfusionMatrixDisplay(
		confusion_matrix=cm, 
		display_labels=kaggle.label_map
	)

	disp.plot()
	disp.ax_.set_title(f"Test Predictions for {args.model} Model")
	plt.savefig(f"fig/confusion-{args.model}.png")  # Save to file
	plt.close()
