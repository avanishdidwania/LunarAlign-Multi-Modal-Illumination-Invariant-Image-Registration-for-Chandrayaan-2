import cv2
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple
from lunar_reg.matching.base import MatchPair

class RobustMethod(Enum):
    RANSAC = "ransac"
    MAGSAC_PLUS = "magsac++"
    LMEDS = "lmeds"

@dataclass
class OutlierRejectionResult:
    inlier_matches: List[MatchPair]
    outlier_matches: List[MatchPair]
    inlier_mask: np.ndarray            # Boolean mask over input matches
    inlier_count: int
    inlier_ratio: float                # inliers / total
    model_matrix: Optional[np.ndarray] # 3x3 matrix representing model (homogeneous)

class OutlierRejector:
    """Robust outlier rejection using RANSAC/MAGSAC++."""

    def __init__(self, method: RobustMethod = RobustMethod.MAGSAC_PLUS,
                 confidence: float = 0.999, max_iterations: int = 10000,
                 threshold: float = 3.0):
        self.method = method
        self.confidence = confidence
        self.max_iterations = max_iterations
        self.threshold = threshold

    def reject(self, matches: List[MatchPair],
               model_type: str = "homography") -> OutlierRejectionResult:
        """
        Apply robust estimation to filter outliers.
        model_type: "homography" | "affine" | "fundamental"
        """
        total = len(matches)
        if total == 0:
            return OutlierRejectionResult(
                inlier_matches=[],
                outlier_matches=[],
                inlier_mask=np.zeros(0, dtype=bool),
                inlier_count=0,
                inlier_ratio=0.0,
                model_matrix=None
            )
            
        src_pts = np.array([m.source_pt for m in matches], dtype=np.float32)
        ref_pts = np.array([m.reference_pt for m in matches], dtype=np.float32)
        
        inlier_mask = np.zeros(total, dtype=bool)
        model_matrix = None
        
        # Determine minimum points required
        min_pts = 4
        if model_type == "affine":
            min_pts = 3
        elif model_type == "fundamental":
            min_pts = 8
            
        if total < min_pts:
            return OutlierRejectionResult(
                inlier_matches=[],
                outlier_matches=list(matches),
                inlier_mask=inlier_mask,
                inlier_count=0,
                inlier_ratio=0.0,
                model_matrix=None
            )
            
        # Determine method flag
        method_flag = cv2.RANSAC
        if self.method == RobustMethod.RANSAC:
            method_flag = cv2.RANSAC
        elif self.method == RobustMethod.MAGSAC_PLUS:
            method_flag = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        elif self.method == RobustMethod.LMEDS:
            method_flag = cv2.LMEDS
            
        try:
            if model_type == "homography":
                H, mask = cv2.findHomography(
                    src_pts, ref_pts,
                    method=method_flag,
                    ransacReprojThreshold=self.threshold,
                    maxIters=self.max_iterations,
                    confidence=self.confidence
                )
                if H is not None:
                    model_matrix = H
                    inlier_mask = (mask.flatten() > 0)
            elif model_type == "affine":
                M, mask = cv2.estimateAffine2D(
                    src_pts, ref_pts,
                    method=method_flag,
                    ransacReprojThreshold=self.threshold,
                    maxIters=self.max_iterations,
                    confidence=self.confidence
                )
                if M is not None:
                    model_matrix = np.eye(3, dtype=np.float64)
                    model_matrix[:2, :] = M
                    inlier_mask = (mask.flatten() > 0)
            elif model_type == "fundamental":
                fm_method = cv2.FM_RANSAC
                if self.method == RobustMethod.LMEDS:
                    fm_method = cv2.FM_LMEDS
                elif self.method == RobustMethod.MAGSAC_PLUS:
                    fm_method = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
                    
                F, mask = cv2.findFundamentalMat(
                    src_pts, ref_pts,
                    method=fm_method,
                    ransacReprojThreshold=self.threshold,
                    confidence=self.confidence,
                    maxIters=self.max_iterations
                )
                if F is not None and F.shape == (3, 3):
                    model_matrix = F
                    inlier_mask = (mask.flatten() > 0)
        except Exception:
            # Catch internal OpenCV failures and degrade gracefully
            pass
            
        # Post-filter inliers to guarantee they satisfy the threshold constraint
        if model_matrix is not None and total > 0:
            try:
                src_hom = np.hstack([src_pts, np.ones((total, 1))])
                proj_hom = src_hom @ model_matrix.T
                proj_pts = proj_hom[:, :2] / (proj_hom[:, 2:3] + 1e-8)
                residuals = np.linalg.norm(proj_pts - ref_pts, axis=1)
                # Keep as inliers only if the solver marked them AND their residual is within threshold
                inlier_mask = inlier_mask & (residuals <= self.threshold)
            except Exception:
                pass
            
        inliers = [matches[i] for i in range(total) if inlier_mask[i]]
        outliers = [matches[i] for i in range(total) if not inlier_mask[i]]
        inlier_count = len(inliers)
        inlier_ratio = float(inlier_count / total) if total > 0 else 0.0
        
        return OutlierRejectionResult(
            inlier_matches=inliers,
            outlier_matches=outliers,
            inlier_mask=inlier_mask,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            model_matrix=model_matrix
        )
