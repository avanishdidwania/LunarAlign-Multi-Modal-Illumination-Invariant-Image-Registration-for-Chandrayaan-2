import cv2
import numpy as np
from typing import Optional
from lunar_reg.detection.base import DetectionResult
from lunar_reg.matching.base import FeatureMatcherBase, MatchPair, MatchingResult

class BFMatcher(FeatureMatcherBase):
    """Brute-force matcher with Lowe's ratio test for SIFT/sparse features."""

    def __init__(self, ratio_threshold: float = 0.8, norm_type: str = "L2"):
        self.ratio_threshold = ratio_threshold
        if norm_type == "L2":
            self.cv_norm = cv2.NORM_L2
        elif norm_type == "HAMMING":
            self.cv_norm = cv2.NORM_HAMMING
        else:
            self.cv_norm = cv2.NORM_L2
            
        self.matcher = cv2.BFMatcher(self.cv_norm)

    def match(self, source_result: DetectionResult,
              reference_result: DetectionResult) -> MatchingResult:
        src_desc = source_result.descriptors
        ref_desc = reference_result.descriptors
        
        if len(src_desc) == 0 or len(ref_desc) == 0:
            return MatchingResult(matches=[])
            
        # If reference has less than 2 descriptors, KNN match with k=2 is impossible
        if len(ref_desc) < 2:
            matches = []
            for i, sd in enumerate(src_desc):
                dists = np.linalg.norm(ref_desc - sd, axis=1)
                idx = int(np.argmin(dists))
                dist = dists[idx]
                confidence = float(1.0 / (1.0 + dist))
                
                src_pt = source_result.keypoints[i]
                ref_pt = reference_result.keypoints[idx]
                
                matches.append(
                    MatchPair(
                        source_idx=i,
                        reference_idx=idx,
                        source_pt=(src_pt.x, src_pt.y),
                        reference_pt=(ref_pt.x, ref_pt.y),
                        confidence=confidence
                    )
                )
            return MatchingResult(matches=matches)
            
        raw_matches = self.matcher.knnMatch(src_desc, ref_desc, k=2)
        
        matches = []
        for m in raw_matches:
            if len(m) == 2:
                m1, m2 = m
                if m1.distance < self.ratio_threshold * m2.distance:
                    ratio = m1.distance / (m2.distance + 1e-8)
                    confidence = float(1.0 - ratio)
                    
                    src_pt = source_result.keypoints[m1.queryIdx]
                    ref_pt = reference_result.keypoints[m1.trainIdx]
                    
                    matches.append(
                        MatchPair(
                            source_idx=m1.queryIdx,
                            reference_idx=m1.trainIdx,
                            source_pt=(src_pt.x, src_pt.y),
                            reference_pt=(ref_pt.x, ref_pt.y),
                            confidence=confidence
                        )
                    )
                    
        return MatchingResult(matches=matches)

    def name(self) -> str:
        return "bf"
