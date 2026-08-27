import ssl
import numpy as np
import torch
from typing import Optional
from kornia.feature import LoFTR
from lunar_reg.detection.base import DetectionResult, Keypoint
from lunar_reg.matching.base import FeatureMatcherBase, MatchPair, MatchingResult
from lunar_reg.device import DeviceManager

# Bypass SSL certificate validation globally for pretrained model downloads
ssl._create_default_https_context = ssl._create_unverified_context

class LoFTRMatcher(FeatureMatcherBase):
    """
    LoFTR detector-free dense matcher - does NOT require precomputed keypoints.
    Uses transformer cross-attention for dense matching, excels in low-texture regions.
    Operates on image pairs directly.
    """

    def __init__(self, pretrained: str = "outdoor", device: str = "auto",
                 match_threshold: float = 0.2):
        self.match_threshold = match_threshold
        
        self.device_manager = DeviceManager(preferred=device)
        self.device = self.device_manager.get_torch_device()
        
        self.model = LoFTR(pretrained=pretrained).to(self.device)
        self.model.eval()

    def match_images(self, source_image: np.ndarray,
                     reference_image: np.ndarray) -> MatchingResult:
        """
        Perform dense matching directly on image pair.
        No prior keypoint detection needed - LoFTR handles end-to-end.
        Returns dense correspondences with confidence scores.
        """
        # Ensure images are float32 normalized to [0, 1]
        if source_image.dtype == np.uint8:
            src_float = source_image.astype(np.float32) / 255.0
        else:
            src_min, src_max = source_image.min(), source_image.max()
            src_float = (source_image - src_min) / (src_max - src_min + 1e-8) if src_max > src_min else np.zeros_like(source_image, dtype=np.float32)
            
        if reference_image.dtype == np.uint8:
            ref_float = reference_image.astype(np.float32) / 255.0
        else:
            ref_min, ref_max = reference_image.min(), reference_image.max()
            ref_float = (reference_image - ref_min) / (ref_max - ref_min + 1e-8) if ref_max > ref_min else np.zeros_like(reference_image, dtype=np.float32)
            
        h0, w0 = src_float.shape[:2]
        h1, w1 = ref_float.shape[:2]
        
        # Format tensors to (1, 1, H, W)
        img0 = torch.tensor(src_float, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        img1 = torch.tensor(ref_float, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        
        data = {
            "image0": img0,
            "image1": img1
        }
        
        with torch.no_grad():
            output = self.model(data)
            
        kpts0 = output["keypoints0"].cpu().numpy()  # (M, 2)
        kpts1 = output["keypoints1"].cpu().numpy()  # (M, 2)
        conf = output["confidence"].cpu().numpy()    # (M,)
        
        matches = []
        source_keypoints = []
        reference_keypoints = []
        
        idx = 0
        for i in range(len(kpts0)):
            score = float(conf[i])
            if score >= self.match_threshold:
                pt0 = (float(kpts0[i, 0]), float(kpts0[i, 1]))
                pt1 = (float(kpts1[i, 2] if kpts1.shape[1] > 2 else kpts1[i, 0]), float(kpts1[i, 3] if kpts1.shape[1] > 2 else kpts1[i, 1]))
                # Check for standard 2D coordinates in kpts1
                pt1_x = float(kpts1[i, 0])
                pt1_y = float(kpts1[i, 1])
                
                matches.append(
                    MatchPair(
                        source_idx=idx,
                        reference_idx=idx,
                        source_pt=pt0,
                        reference_pt=(pt1_x, pt1_y),
                        confidence=score
                    )
                )
                
                source_keypoints.append(
                    Keypoint(x=pt0[0], y=pt0[1], scale=1.0, orientation=0.0, response=score)
                )
                reference_keypoints.append(
                    Keypoint(x=pt1_x, y=pt1_y, scale=1.0, orientation=0.0, response=score)
                )
                idx += 1
                
        return MatchingResult(
            matches=matches,
            source_keypoints=source_keypoints,
            reference_keypoints=reference_keypoints
        )

    def match(self, source_result: DetectionResult,
              reference_result: DetectionResult) -> MatchingResult:
        """
        Fallback implementation to satisfy FeatureMatcherBase.
        Since LoFTR is detector-free, this is not the preferred execution path.
        """
        raise NotImplementedError(
            "LoFTR requires image inputs directly. Use match_images(source_image, reference_image) instead."
        )

    def name(self) -> str:
        return "loftr"
