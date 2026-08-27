import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
import cv2
import pytest
from lunar_reg.refinement.subpixel import SubPixelRefiner
from lunar_reg.matching.base import MatchPair
from tests.conftest import random_grayscale_image

# Feature: lunar-image-registration, Property 13: Sub-pixel refinement precision
# Validates: Requirements 7.1
@given(
    dx=st.floats(-0.6, 0.6),
    dy=st.floats(-0.6, 0.6)
)
@settings(max_examples=20, deadline=None)
def test_subpixel_refinement_precision(dx, dy):
    refiner = SubPixelRefiner(patch_size=15, min_correlation=0.7)
    
    # Construct a smooth sinusoidal image
    h, w = 128, 128
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    # 2D sine wave pattern with period of 16 pixels
    base_img = 127.5 + 127.5 * np.sin(2.0 * np.pi * x_coords / 16.0) * np.sin(2.0 * np.pi * y_coords / 16.0)
    base_img = base_img.astype(np.uint8)
    
    # Warp with known sub-pixel shift
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    shifted_img = cv2.warpAffine(base_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Select a feature point near the center where boundaries are safe
    src_x, src_y = 64.0, 64.0
    
    # Integer approximation of matched reference point
    approx_ref_x = float(round(src_x + dx))
    approx_ref_y = float(round(src_y + dy))
    
    matches = [
        MatchPair(
            source_idx=0,
            reference_idx=0,
            source_pt=(src_x, src_y),
            reference_pt=(approx_ref_x, approx_ref_y),
            confidence=0.5
        )
    ]
    
    refined = refiner.refine_matches(base_img, shifted_img, matches)
    
    # Check that the shift was recovered with high precision (error <= 0.05 pixels)
    assert len(refined) == 1
    m_ref = refined[0]
    
    est_dx = m_ref.reference_pt[0] - src_x
    est_dy = m_ref.reference_pt[1] - src_y
    
    assert abs(est_dx - dx) <= 0.05, f"dx error: {abs(est_dx - dx)}"
    assert abs(est_dy - dy) <= 0.05, f"dy error: {abs(est_dy - dy)}"

# Feature: lunar-image-registration, Property 14: Sub-pixel outlier rejection
# Validates: Requirements 7.3
@given(
    min_corr=st.floats(0.8, 0.95)
)
@settings(max_examples=15, deadline=None)
def test_subpixel_outlier_rejection(min_corr):
    refiner = SubPixelRefiner(patch_size=15, min_correlation=min_corr)
    
    # Generate two completely unrelated random noise images
    h, w = 128, 128
    img_src = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    img_ref = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    
    # Random keypoint matches
    matches = [
        MatchPair(
            source_idx=0,
            reference_idx=0,
            source_pt=(64.0, 64.0),
            reference_pt=(64.0, 64.0),
            confidence=0.5
        )
    ]
    
    refined = refiner.refine_matches(img_src, img_ref, matches)
    
    # Since the images are unrelated noise, correlation should be low and rejected
    assert len(refined) == 0
