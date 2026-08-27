import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from kornia.feature import LightGlue
from lunar_reg.detection.base import DetectionResult, Keypoint
from lunar_reg.matching.base import FeatureMatcherBase, MatchPair, MatchingResult
from lunar_reg.device import DeviceManager

logger = logging.getLogger(__name__)

class LightGlueMatcher(FeatureMatcherBase):
    """
    LightGlue sparse matcher - pairs with SuperPoint detections.
    Uses adaptive pruning for efficiency while maintaining accuracy.
    """

    def __init__(self, features: str = "superpoint", device: str = "auto",
                 match_threshold: float = 0.2, depth_confidence: float = 0.95,
                 width_confidence: float = 0.99):
        self.features = features
        self.match_threshold = match_threshold
        
        self.device_manager = DeviceManager(preferred=device)
        self.device = self.device_manager.get_torch_device()
        
        self.model = LightGlue(
            features=features,
            depth_confidence=depth_confidence,
            width_confidence=width_confidence
        ).to(self.device)
        self.model.eval()

    def match(self, source_result: DetectionResult,
              reference_result: DetectionResult) -> MatchingResult:
        src_kps = source_result.keypoints
        ref_kps = reference_result.keypoints
        
        # LightGlue requires at least 2 keypoints in each image to match
        if len(src_kps) < 2 or len(ref_kps) < 2:
            return MatchingResult(matches=[])
            
        src_h, src_w = source_result.image_shape
        ref_h, ref_w = reference_result.image_shape
        
        # Prepare tensors in original pixel coordinates
        src_pts = np.array([[kp.x, kp.y] for kp in src_kps], dtype=np.float32)
        ref_pts = np.array([[kp.x, kp.y] for kp in ref_kps], dtype=np.float32)
        
        # Convert to torch
        kpts0 = torch.tensor(src_pts, dtype=torch.float32, device=self.device).unsqueeze(0)
        kpts1 = torch.tensor(ref_pts, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        desc0 = torch.tensor(source_result.descriptors, dtype=torch.float32, device=self.device).unsqueeze(0)
        desc1 = torch.tensor(reference_result.descriptors, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        size0 = torch.tensor([[float(src_w), float(src_h)]], dtype=torch.float32, device=self.device)
        size1 = torch.tensor([[float(ref_w), float(ref_h)]], dtype=torch.float32, device=self.device)
        
        data = {
            "image0": {
                "keypoints": kpts0,
                "descriptors": desc0,
                "image_size": size0
            },
            "image1": {
                "keypoints": kpts1,
                "descriptors": desc1,
                "image_size": size1
            }
        }
        
        try:
            with torch.no_grad():
                output = self.model(data)
                
                # matches is shape (M, 2)
                matches_tensor = output["matches"][0]
                scores_tensor = output["scores"][0]
                
            matches_np = matches_tensor.cpu().numpy()
            scores_np = scores_tensor.cpu().numpy()
        except Exception as e:
            logger.warning(f"LightGlue internal model forward pass failed: {e}")
            return MatchingResult(matches=[])
        
        matches = []
        for i in range(len(matches_np)):
            idx0, idx1 = matches_np[i]
            score = float(scores_np[i])
            
            if score >= self.match_threshold:
                matches.append(
                    MatchPair(
                        source_idx=int(idx0),
                        reference_idx=int(idx1),
                        source_pt=(src_kps[idx0].x, src_kps[idx0].y),
                        reference_pt=(ref_kps[idx1].x, ref_kps[idx1].y),
                        confidence=score
                    )
                )
                
        return MatchingResult(matches=matches)

    def name(self) -> str:
        return "lightglue"
