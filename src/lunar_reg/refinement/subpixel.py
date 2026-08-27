import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Optional
from lunar_reg.matching.base import MatchPair

logger = logging.getLogger(__name__)

@dataclass
class RefinedMatch:
    source_pt: tuple[float, float]       # Sub-pixel refined (x, y) in source
    reference_pt: tuple[float, float]    # Sub-pixel refined (x, y) in reference
    accuracy_estimate: float             # Estimated accuracy in pixels
    ncc_score: float                     # Normalized cross-correlation at refined position

@dataclass
class RefinementResult:
    refined_matches: List[RefinedMatch]
    mean_accuracy: float
    median_accuracy: float

class SubPixelRefiner:
    """
    Sub-pixel alignment refiner using Normalized Cross-Correlation (NCC)
    and quadratic surface interpolation.
    Refines pixel-level matches to sub-pixel accuracy.
    """

    def __init__(self, patch_size: int = 15, min_correlation: float = 0.8, search_radius: int = 3):
        self.patch_size = patch_size
        self.min_correlation = min_correlation
        self.search_radius = search_radius

    def refine_matches(self, source_image: np.ndarray,
                       reference_image: np.ndarray,
                       matches: List[MatchPair]) -> List[MatchPair]:
        """
        Refine the keypoint coordinates of matches to sub-pixel precision.
        Matches with peak correlation below min_correlation are discarded.
        """
        refined_matches = []
        r = self.search_radius
        w = self.patch_size // 2
        
        h_src, w_src = source_image.shape[:2]
        h_ref, w_ref = reference_image.shape[:2]
        
        for m in matches:
            # Round source and reference coordinates to nearest integers
            ix_src, iy_src = int(round(m.source_pt[0])), int(round(m.source_pt[1]))
            ix_ref, iy_ref = int(round(m.reference_pt[0])), int(round(m.reference_pt[1]))
            
            # Check boundaries for source patch
            if (ix_src - w < 0 or ix_src + w >= w_src or
                iy_src - w < 0 or iy_src + w >= h_src):
                continue
                
            # Check boundaries for reference search window
            if (ix_ref - w - r < 0 or ix_ref + w + r >= w_ref or
                iy_ref - w - r < 0 or iy_ref + w + r >= h_ref):
                continue
                
            # Extract patches
            src_patch = source_image[iy_src - w : iy_src + w + 1, ix_src - w : ix_src + w + 1]
            ref_window = reference_image[iy_ref - w - r : iy_ref + w + r + 1, ix_ref - w - r : ix_ref + w + r + 1]
            
            # Convert to float32 unconditionally (OpenCV matchTemplate only supports uint8 or float32)
            src_patch = src_patch.astype(np.float32)
            ref_window = ref_window.astype(np.float32)
                
            # Check for zero variance patches to avoid division by zero in template matching
            if np.std(src_patch) < 1e-4 or np.std(ref_window) < 1e-4:
                continue
                
            # Compute NCC map
            ncc_map = cv2.matchTemplate(ref_window, src_patch, cv2.TM_CCOEFF_NORMED)
            
            # Find peak
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(ncc_map)
            px, py = max_loc  # col, row of peak in ncc_map
            
            if max_val < self.min_correlation:
                continue
                
            dx = 0.0
            dy = 0.0
            
            # Perform quadratic interpolation if the peak is in the interior
            ny, nx = ncc_map.shape
            if 0 < px < nx - 1:
                c_prev = float(ncc_map[py, px - 1])
                c_curr = float(ncc_map[py, px])
                c_next = float(ncc_map[py, px + 1])
                denom = 2.0 * c_curr - c_prev - c_next
                if abs(denom) > 1e-5:
                    dx = float((c_next - c_prev) / (2.0 * denom))
                    
            if 0 < py < ny - 1:
                r_prev = float(ncc_map[py - 1, px])
                r_curr = float(ncc_map[py, px])
                r_next = float(ncc_map[py + 1, px])
                denom = 2.0 * r_curr - r_prev - r_next
                if abs(denom) > 1e-5:
                    dy = float((r_next - r_prev) / (2.0 * denom))
                    
            # Compute refined reference coordinate
            refined_x_ref = float(ix_ref - r + px + dx)
            refined_y_ref = float(iy_ref - r + py + dy)
            
            # Keep source point and refine reference point coordinate
            refined_matches.append(
                MatchPair(
                    source_idx=m.source_idx,
                    reference_idx=m.reference_idx,
                    source_pt=m.source_pt,
                    reference_pt=(refined_x_ref, refined_y_ref),
                    confidence=float(max_val)
                )
            )
            
        return refined_matches

    def refine(self, source_image: np.ndarray,
               reference_image: np.ndarray,
               matches: List[MatchPair]) -> RefinementResult:
        """
        Refine the keypoint coordinates of matches to sub-pixel precision and return RefinementResult.
        """
        refined_pairs = self.refine_matches(source_image, reference_image, matches)
        
        refined_matches = []
        accuracies = []
        for m in refined_pairs:
            # Estimate accuracy in pixels based on correlation value
            # Standard formula: accuracy = 0.05 * (1.0 - correlation) + 0.01
            ncc = m.confidence
            accuracy = float(0.05 * (1.0 - ncc) + 0.01)
            accuracies.append(accuracy)
            
            refined_matches.append(
                RefinedMatch(
                    source_pt=m.source_pt,
                    reference_pt=m.reference_pt,
                    accuracy_estimate=accuracy,
                    ncc_score=ncc
                )
            )
            
        mean_acc = float(np.mean(accuracies)) if accuracies else 0.0
        med_acc = float(np.median(accuracies)) if accuracies else 0.0
        
        return RefinementResult(
            refined_matches=refined_matches,
            mean_accuracy=mean_acc,
            median_accuracy=med_acc
        )
