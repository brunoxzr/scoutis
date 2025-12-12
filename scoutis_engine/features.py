from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import cv2

class FeatureExtractor:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        m = models.mobilenet_v3_small(weights=weights)
        m.classifier = nn.Identity()
        m.eval().to(self.device)
        self.model = m
        self.transform = weights.transforms()

    @torch.no_grad()
    def embed(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2,0,1).to(torch.uint8)
        x = self.transform(x).unsqueeze(0).to(self.device)
        feat = self.model(x).squeeze(0).detach().cpu().numpy().astype(np.float32)
        return feat
