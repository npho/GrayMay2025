"""
BrainTumorNet.py (May 2025)
"""

import torch
import torch.nn as nn

class BrainTumorNet(torch.nn.Module):
	"""
	A simple neural network architecture for brain tumor classification.
	"""

	def __init__(self):
		super(BrainTumorNet, self).__init__()
		
		self.cnn1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=8, kernel_size=12, stride=2, padding=2),
			nn.ReLU(),
			nn.AvgPool2d(kernel_size=12, stride=2, padding=2)
		)
		
		self.cnn2 = nn.Sequential(
			nn.Conv2d(in_channels=8, out_channels=128, kernel_size=12, stride=2, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=12, stride=2, padding=2)
		)
		
		l = 500
		self.fc1 = nn.Sequential(
			nn.Linear(86528, l),
			nn.ReLU(),
			nn.Linear(l, 4)
		)

	def forward(self, x):
		out = self.cnn1(x)
		out = self.cnn2(out)
		out = out.view(out.size(0), -1)		
		out = self.fc1(out)
		
		return out
