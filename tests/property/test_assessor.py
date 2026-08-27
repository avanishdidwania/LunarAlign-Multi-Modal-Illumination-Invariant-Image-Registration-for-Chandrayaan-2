import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
import cv2
import pytest
from lunar_reg.evaluation.assessor import QualityAssessor
from tests.conftest import random_grayscale_image

# Feature: lunar-image-registration, Property 17: Quality score identity
# Validates: Requirements 9.1
@given(
    image=random_grayscale_image(min_size=128, max_size=256)
)
@settings(max_examples=15, deadline=None)
def test_quality_score_identity(image):
    assessor = QualityAssessor()
    
    # Ensure some texture and non-zero pixels so that overlap exists
    if np.sum(image > 0) < 150:
        h, w = image.shape
        image = np.zeros((h, w), dtype=np.uint8)
        for y in range(0, h, 8):
            for x in range(0, w, 8):
                if ((x // 8) + (y // 8)) % 2 == 0:
                    image[y:y+8, x:x+8] = 255
        
    metrics = assessor.assess(image, image, inlier_matches=[], total_initial_matches=0)
    
    assert abs(metrics.ssim - 1.0) < 1e-3
    assert abs(metrics.mutual_information - 1.0) < 1e-3
    assert metrics.psnr == float('inf')
    
    # Check that when matches are also perfect, q_score is 1.0
    from lunar_reg.matching.base import MatchPair
    dummy_matches = [MatchPair(i, i, (0.0, 0.0), (0.0, 0.0), 1.0) for i in range(10)]
    metrics_perf = assessor.assess(image, image, inlier_matches=dummy_matches, total_initial_matches=10)
    assert abs(metrics_perf.q_score - 1.0) < 1e-3

# Feature: lunar-image-registration, Property 18: Quality score range bounds
# Validates: Requirements 9.4
@given(
    img1=random_grayscale_image(min_size=128, max_size=256),
    img2=random_grayscale_image(min_size=128, max_size=256),
    inliers_cnt=st.integers(0, 20),
    total_cnt=st.integers(0, 40)
)
@settings(max_examples=25, deadline=None)
def test_quality_score_range_bounds(img1, img2, inliers_cnt, total_cnt):
    assessor = QualityAssessor()
    
    if inliers_cnt > total_cnt:
        total_cnt = inliers_cnt
        
    # Crop/resize img2 to match img1 shape for testing
    if img2.shape != img1.shape:
        h, w = img1.shape
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_NEAREST)
        
    from lunar_reg.matching.base import MatchPair
    dummy_inliers = [MatchPair(i, i, (0.0, 0.0), (0.0, 0.0), 1.0) for i in range(inliers_cnt)]
    
    metrics = assessor.assess(img1, img2, inlier_matches=dummy_inliers, total_initial_matches=total_cnt)
    
    assert 0.0 <= metrics.ssim <= 1.0
    assert 0.0 <= metrics.mutual_information <= 1.0
    assert 0.0 <= metrics.inlier_ratio <= 1.0
    assert 0.0 <= metrics.q_score <= 1.0
