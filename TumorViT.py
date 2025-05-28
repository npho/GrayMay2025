"""
TumorViT.py (May 2025)
"""

import torch
import torch.nn as nn
from transformers import ViTModel

class TumorViT(torch.nn.Module):

    def __init__(self, vit_path, freeze_vit=True):
        super(TumorViT, self).__init__()
        self.vit = ViTModel.from_pretrained(vit_path)
        if freeze_vit:
            self.vit.requires_grad_(False)  # <-- Freeze backbone

        hidden_dim = self.vit.config.hidden_size
        self.fc = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        outputs = self.vit(x)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.fc(cls_token)
