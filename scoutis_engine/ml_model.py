from __future__ import annotations
import torch
import torch.nn as nn

class EmbeddingAutoEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        h = max(64, dim // 4)
        z = max(32, dim // 8)
        self.net = nn.Sequential(
            nn.Linear(dim, h), nn.ReLU(),
            nn.Linear(h, z), nn.ReLU(),
            nn.Linear(z, h), nn.ReLU(),
            nn.Linear(h, dim),
        )

    def forward(self, x):
        return self.net(x)
