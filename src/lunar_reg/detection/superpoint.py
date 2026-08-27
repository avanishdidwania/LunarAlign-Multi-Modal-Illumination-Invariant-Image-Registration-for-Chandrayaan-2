import os
import urllib.request
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import maximum_filter
from lunar_reg.detection.base import FeatureDetectorBase, Keypoint, DetectionResult
from lunar_reg.device import DeviceManager

logger = logging.getLogger(__name__)

SUPERPOINT_WEIGHTS_URL = "https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth"

class SuperPointNet(nn.Module):
    """SuperPoint network architecture mapping to the pretrained weights."""
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Shared Encoder
        self.conv1a = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Detector Head
        self.convPa = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)
        
        # Descriptor Head
        self.convDa = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Shared Encoder
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        
        # Detector Head
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        
        # Descriptor Head
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1, keepdim=True)
        desc = desc / (dn + 1e-8)  # L2 normalize
        
        return semi, desc

class SuperPointDetector(FeatureDetectorBase):
    """SuperPoint deep learning feature detector (MagicLeap)."""

    def __init__(self, max_keypoints: int = 4096, keypoint_threshold: float = 0.005,
                 nms_dist: int = 4, device: str = "auto", weights_path: Optional[Path] = None):
        self.max_keypoints = max_keypoints
        self.keypoint_threshold = keypoint_threshold
        self.nms_dist = nms_dist
        
        # Resolve device
        self.device_manager = DeviceManager(preferred=device)
        self.device = self.device_manager.get_torch_device()
        
        # Path to weights
        if weights_path is None:
            # Default to a models directory in the project
            self.weights_path = Path(__file__).resolve().parents[3] / "models" / "superpoint_v1.pth"
        else:
            self.weights_path = weights_path
            
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        """Load SuperPoint model with pretrained weights."""
        if not self.weights_path.exists():
            try:
                self.weights_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Downloading SuperPoint weights to {self.weights_path}...")
                urllib.request.urlretrieve(SUPERPOINT_WEIGHTS_URL, self.weights_path)
            except Exception as e:
                logger.error(f"Failed to download SuperPoint weights: {e}")
                raise RuntimeError(f"Failed to load SuperPoint weights: {e}") from e
                
        model = SuperPointNet()
        try:
            # Load state dict
            state_dict = torch.load(self.weights_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            logger.info("SuperPoint model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load SuperPoint weights from file: {e}")
            raise RuntimeError(f"Failed to instantiate SuperPoint model: {e}") from e

    def detect(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> DetectionResult:
        """Run SuperPoint inference. Returns keypoints with 256-dim descriptors."""
        # Ensure image is 2D grayscale float32 normalized to [0, 1]
        if image.dtype == np.uint8:
            img_float = image.astype(np.float32) / 255.0
        else:
            # Clip or scale to [0, 1]
            img_min = image.min()
            img_max = image.max()
            if img_max > img_min:
                img_float = (image - img_min) / (img_max - img_min)
            else:
                img_float = np.zeros_like(image, dtype=np.float32)
                
        h, w = img_float.shape[:2]
        
        # Prepare input tensor: shape (1, 1, H, W)
        inp = torch.tensor(img_float, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            semi, desc = self.model(inp)
            
            # 1. Process detector heatmap
            semi = F.softmax(semi, dim=1)
            nodust = semi[:, :64, :, :]  # Shape: (1, 64, H/8, W/8)
            
            # Reshape cell grid to spatial dimensions
            B, _, H_c, W_c = nodust.shape
            nodust = nodust.permute(0, 2, 3, 1)  # (1, H_c, W_c, 64)
            nodust = nodust.view(B, H_c, W_c, 8, 8)
            nodust = nodust.permute(0, 1, 3, 2, 4)  # (1, H_c, 8, W_c, 8)
            heatmap = nodust.reshape(H_c * 8, W_c * 8).cpu().numpy()
            
        # Apply mask if available
        if mask is not None:
            heatmap = heatmap * (mask > 0)
            
        # 2. Extract keypoints via Non-Maximum Suppression (NMS)
        local_max = (heatmap == maximum_filter(heatmap, size=2*self.nms_dist+1))
        coords = np.argwhere((heatmap >= self.keypoint_threshold) & local_max)  # shape (N, 2) with (y, x)
        
        if len(coords) == 0:
            return DetectionResult(
                keypoints=[],
                descriptors=np.zeros((0, 256), dtype=np.float32),
                image_shape=(h, w)
            )
            
        # Get scores
        scores = heatmap[coords[:, 0], coords[:, 1]]
        
        # Sort by score in descending order
        indices = np.argsort(scores)[::-1]
        coords = coords[indices]
        scores = scores[indices]
        
        # Limit to max keypoints
        if len(coords) > self.max_keypoints:
            coords = coords[:self.max_keypoints]
            scores = scores[:self.max_keypoints]
            
        # Convert coords to internal Keypoint model
        keypoints = []
        for i in range(len(coords)):
            y, x = coords[i]
            keypoints.append(
                Keypoint(
                    x=float(x),
                    y=float(y),
                    scale=1.0,  # SuperPoint has no multi-scale octave scale
                    orientation=0.0,
                    response=float(scores[i])
                )
            )
            
        # 3. Sample descriptors using bilinear interpolation
        # grid_sample expects normalized coordinates in [-1, 1] and in (x, y) format
        pts_torch = torch.tensor(coords, dtype=torch.float32, device=self.device)  # (N, 2) as (y, x)
        grid_coords = pts_torch[:, [1, 0]].clone()  # (N, 2) as (x, y)
        grid_coords[:, 0] = 2.0 * grid_coords[:, 0] / (w - 1) - 1.0
        grid_coords[:, 1] = 2.0 * grid_coords[:, 1] / (h - 1) - 1.0
        grid_coords = grid_coords.view(1, 1, -1, 2)  # Shape (1, 1, N, 2)
        
        with torch.no_grad():
            desc_sampled = F.grid_sample(desc, grid_coords, mode="bilinear", padding_mode="zeros", align_corners=True)
            desc_sampled = desc_sampled.view(256, -1).t()  # (N, 256)
            # Re-normalize to unit length
            desc_sampled = desc_sampled / (torch.norm(desc_sampled, p=2, dim=1, keepdim=True) + 1e-8)
            descriptors = desc_sampled.cpu().numpy()
            
        return DetectionResult(
            keypoints=keypoints,
            descriptors=descriptors,
            image_shape=(h, w)
        )

    def name(self) -> str:
        return "superpoint"
