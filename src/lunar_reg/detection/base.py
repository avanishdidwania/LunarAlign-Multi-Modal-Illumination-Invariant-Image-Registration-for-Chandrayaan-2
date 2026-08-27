from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Keypoint:
    x: float
    y: float
    scale: float
    orientation: float  # radians
    response: float     # detector response strength

@dataclass
class DetectionResult:
    keypoints: List[Keypoint]
    descriptors: np.ndarray      # Shape: (N, descriptor_dim)
    image_shape: tuple[int, int]

class FeatureDetectorBase(ABC):
    """Abstract interface for feature detection algorithms."""

    @abstractmethod
    def detect(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> DetectionResult:
        """Detect keypoints and compute descriptors."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Return algorithm name for logging/config."""
        pass

    def spatial_distribution_score(self, keypoints: List[Keypoint],
                                    image_shape: tuple[int, int],
                                    grid_size: int = 8) -> float:
        """
        Compute how uniformly distributed keypoints are across the image.
        Divides image into grid_size x grid_size cells, measures coefficient
        of variation of keypoint counts per cell.
        Returns score in [0, 1] where 1 = perfectly uniform.
        """
        if not keypoints:
            return 0.0
        
        h, w = image_shape
        grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        
        cell_w = w / grid_size
        cell_h = h / grid_size
        
        for kp in keypoints:
            cx = int(kp.x // cell_w)
            cy = int(kp.y // cell_h)
            
            # clip to bounds in case coordinates are exactly on boundary
            cx = min(max(cx, 0), grid_size - 1)
            cy = min(max(cy, 0), grid_size - 1)
            
            grid[cy, cx] += 1
            
        counts = grid.flatten()
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        if mean_count == 0.0:
            return 0.0
            
        cv = std_count / mean_count
        score = max(0.0, 1.0 - cv)
        return float(score)
