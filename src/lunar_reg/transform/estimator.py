import cv2
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple
from lunar_reg.matching.base import MatchPair

class TransformationType(Enum):
    AFFINE = "affine"           # 6 DOF - translation, rotation, scale, shear
    PROJECTIVE = "projective"   # 8 DOF - full homography

@dataclass
class TransformationResult:
    matrix: np.ndarray                  # 3x3 transformation matrix
    transformation_type: TransformationType
    residuals: np.ndarray               # Per-point residual errors
    rmse: float                         # Root mean square reprojection error
    condition_number: float             # Numerical stability indicator (ratio of singular values)

class TransformationEstimator:
    """Estimates geometric transformation from validated match points."""

    def auto_select_model(self, matches: List[MatchPair]) -> TransformationType:
        """
        Select affine vs projective based on spatial extent of match points.
        If the maximum coordinate span of points is >= 2000px, we select projective.
        Otherwise, we select affine.
        """
        if not matches:
            return TransformationType.AFFINE
            
        src_pts = np.array([m.source_pt for m in matches])
        min_coords = np.min(src_pts, axis=0)
        max_coords = np.max(src_pts, axis=0)
        span = max_coords - min_coords
        max_span = float(np.max(span))
        
        if max_span >= 2000.0:
            return TransformationType.PROJECTIVE
        else:
            return TransformationType.AFFINE

    def estimate(self, inlier_matches: List[MatchPair],
                 transformation_type: Optional[TransformationType] = None) -> TransformationResult:
        """
        Estimate transformation from inlier correspondences using least squares.
        """
        total = len(inlier_matches)
        
        if transformation_type is None:
            transformation_type = self.auto_select_model(inlier_matches)
            
        min_pts = 3 if transformation_type == TransformationType.AFFINE else 4
        if total < min_pts:
            raise ValueError(
                f"Insufficient points for {transformation_type.value} estimation. "
                f"Required: {min_pts}, got: {total}"
            )
            
        src_pts = np.array([m.source_pt for m in inlier_matches], dtype=np.float32)
        ref_pts = np.array([m.reference_pt for m in inlier_matches], dtype=np.float32)
        
        if transformation_type == TransformationType.AFFINE:
            # Solve least squares for 6 DOF affine transform:
            # xp = a00 * x + a01 * y + a02
            # yp = a10 * x + a11 * y + a12
            A_lst = []
            b_lst = []
            for m in inlier_matches:
                x, y = m.source_pt
                xp, yp = m.reference_pt
                A_lst.append([x, y, 1, 0, 0, 0])
                A_lst.append([0, 0, 0, x, y, 1])
                b_lst.append(xp)
                b_lst.append(yp)
            A_mat = np.array(A_lst, dtype=np.float64)
            b_mat = np.array(b_lst, dtype=np.float64)
            
            p, _, _, _ = np.linalg.lstsq(A_mat, b_mat, rcond=None)
            matrix = np.eye(3, dtype=np.float64)
            matrix[0, :] = p[:3]
            matrix[1, :] = p[3:]
        else:
            H, _ = cv2.findHomography(src_pts, ref_pts, method=0)  # Least squares DLT
            if H is None:
                raise ValueError("Least-squares homography estimation failed.")
            matrix = H.astype(np.float64)
            
        # Compute residuals
        # Project source points: X = matrix * pt
        src_hom = np.hstack([src_pts, np.ones((total, 1))])
        proj_hom = src_hom @ matrix.T
        proj_pts = proj_hom[:, :2] / (proj_hom[:, 2:3] + 1e-8)
        
        residuals = np.linalg.norm(proj_pts - ref_pts, axis=1)
        rmse = float(np.sqrt(np.mean(residuals**2)))
        
        # Compute condition number of upper-left 2x2 block
        s = np.linalg.svd(matrix[:2, :2], compute_uv=False)
        condition_number = float(s[0] / (s[1] + 1e-8))
        
        return TransformationResult(
            matrix=matrix,
            transformation_type=transformation_type,
            residuals=residuals,
            rmse=rmse,
            condition_number=condition_number
        )

    def validate_transform(self, matrix: np.ndarray) -> bool:
        """
        Validate that transformation is physically plausible:
        - Determinant > 0 (non-degenerate/no mirroring)
        - Scale change within bounds [0.1, 10.0]
        - No extreme shear (condition number <= 10.0)
        """
        det = np.linalg.det(matrix[:2, :2])
        if det <= 0:
            return False
            
        s = np.linalg.svd(matrix[:2, :2], compute_uv=False)
        if s[0] > 10.0 or s[1] < 0.1:
            return False
            
        cond = s[0] / (s[1] + 1e-8)
        if cond > 10.0:
            return False
            
        return True
