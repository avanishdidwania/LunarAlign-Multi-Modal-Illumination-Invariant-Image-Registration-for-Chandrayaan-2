import cv2
import numpy as np
from dataclasses import dataclass
from typing import List
from lunar_reg.matching.base import MatchPair

@dataclass
class QualityMetrics:
    ssim: float
    psnr: float
    mutual_information: float  # Normalized mutual information (NMI) in [0, 1]
    inlier_ratio: float
    q_score: float
    rmse: float
    spatial_distribution_score: float

class QualityAssessor:
    """Evaluates the registration quality of warped images against reference images."""

    def __init__(self, w_ssim: float = 0.4, w_nmi: float = 0.4, w_inlier: float = 0.2):
        self.w_ssim = w_ssim
        self.w_nmi = w_nmi
        self.w_inlier = w_inlier

    def assess(self, warped_source: np.ndarray,
               reference_image: np.ndarray,
               inlier_matches: List[MatchPair],
               total_initial_matches: int,
               rmse: float = 0.0) -> QualityMetrics:
        """
        Assess registration quality over overlapping region.
        """
        # 1. Inlier ratio
        inlier_ratio = 0.0
        if total_initial_matches > 0:
            inlier_ratio = float(len(inlier_matches) / total_initial_matches)
            
        # Identify overlapping region (where both are valid non-zero pixels)
        overlap = (warped_source > 0) & (reference_image > 0)
        overlap_count = np.sum(overlap)
        
        if overlap_count < 100:
            return QualityMetrics(
                ssim=0.0,
                psnr=0.0,
                mutual_information=0.0,
                inlier_ratio=inlier_ratio,
                q_score=0.0,
                rmse=rmse,
                spatial_distribution_score=0.0
            )
            
        val_src = warped_source[overlap].astype(np.float32)
        val_ref = reference_image[overlap].astype(np.float32)
        
        # 2. PSNR
        mse = np.mean((val_src - val_ref)**2)
        if mse < 1e-10:
            psnr = float('inf')
        else:
            # Standard PSNR assuming uint8 range [0, 255]
            psnr = float(20.0 * np.log10(255.0 / np.sqrt(mse)))
            
        # 3. Masked SSIM
        C1 = (0.01 * 255.0)**2
        C2 = (0.03 * 255.0)**2
        
        img1 = warped_source.astype(np.float32)
        img2 = reference_image.astype(np.float32)
        
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(img1**2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2**2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        
        sigma1_sq = np.maximum(sigma1_sq, 0.0)
        sigma2_sq = np.maximum(sigma2_sq, 0.0)
        
        num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = num / (den + 1e-12)
        
        ssim = float(np.mean(ssim_map[overlap]))
        ssim = float(np.clip(ssim, 0.0, 1.0))
        
        # 4. Normalized Mutual Information (NMI)
        # Use a joint histogram of 32x32 bins
        hist_2d, _, _ = np.histogram2d(val_src, val_ref, bins=32)
        pxy = hist_2d / (np.sum(hist_2d) + 1e-12)
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        
        hx = -np.sum(px * np.log2(px + 1e-12))
        hy = -np.sum(py * np.log2(py + 1e-12))
        hxy = -np.sum(pxy * np.log2(pxy + 1e-12))
        
        mi = hx + hy - hxy
        nmi = 2.0 * mi / (hx + hy + 1e-12)
        nmi = float(np.clip(nmi, 0.0, 1.0))
        
        # 5. Combined Q-Score
        q_score = float(self.w_ssim * ssim + self.w_nmi * nmi + self.w_inlier * inlier_ratio)
        q_score = float(np.clip(q_score, 0.0, 1.0))
        
        # 6. Spatial distribution score
        sds = self.compute_spatial_distribution_score(inlier_matches, reference_image.shape)
        
        return QualityMetrics(
            ssim=ssim,
            psnr=psnr,
            mutual_information=nmi,
            inlier_ratio=inlier_ratio,
            q_score=q_score,
            rmse=rmse,
            spatial_distribution_score=sds
        )

    def compute_spatial_distribution_score(self, matches: List[MatchPair], image_shape: tuple, grid_size: int = 8) -> float:
        """Computes grid-based spatial distribution of tie-points in [0, 1]."""
        if not matches:
            return 0.0
        h, w = image_shape[:2]
        grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
        for m in matches:
            rx, ry = m.reference_pt
            bin_x = int(np.floor(rx / (w / grid_size)))
            bin_y = int(np.floor(ry / (h / grid_size)))
            # Clip bins to valid grid coordinates
            bin_x = np.clip(bin_x, 0, grid_size - 1)
            bin_y = np.clip(bin_y, 0, grid_size - 1)
            grid[bin_y, bin_x] = 1
        return float(np.sum(grid) / (grid_size * grid_size))
