import cv2
import numpy as np
from typing import Optional
from lunar_reg.detection.base import FeatureDetectorBase, Keypoint, DetectionResult

class SIFTDetector(FeatureDetectorBase):
    """Classical SIFT detector via OpenCV."""

    def __init__(self, n_features: int = 10000, n_octave_layers: int = 3,
                 contrast_threshold: float = 0.04, edge_threshold: float = 10.0):
        self.n_features = n_features
        self.n_octave_layers = n_octave_layers
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self._sift = cv2.SIFT_create(
            nfeatures=n_features,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
        )

    def detect(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> DetectionResult:
        """Detect SIFT keypoints and compute 128-dim descriptors."""
        # Ensure image is uint8 grayscale
        if image.dtype != np.uint8:
            img_min = image.min()
            img_max = image.max()
            if img_max > img_min:
                img_uint8 = ((image - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
            else:
                img_uint8 = np.zeros_like(image, dtype=np.uint8)
        else:
            img_uint8 = image
            
        kps, desc = self._sift.detectAndCompute(img_uint8, mask)
        
        if kps is None or len(kps) == 0:
            return DetectionResult(
                keypoints=[],
                descriptors=np.zeros((0, 128), dtype=np.float32),
                image_shape=image.shape[:2]
            )
            
        keypoints = []
        for k in kps:
            keypoints.append(
                Keypoint(
                    x=float(k.pt[0]),
                    y=float(k.pt[1]),
                    scale=float(k.size),
                    orientation=float(k.angle * np.pi / 180.0),  # convert to radians
                    response=float(k.response)
                )
            )
            
        return DetectionResult(
            keypoints=keypoints,
            descriptors=desc,
            image_shape=image.shape[:2]
        )

    def name(self) -> str:
        return "sift"
