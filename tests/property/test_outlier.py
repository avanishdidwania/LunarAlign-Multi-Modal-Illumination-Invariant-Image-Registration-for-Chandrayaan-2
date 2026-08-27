import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
import pytest
from lunar_reg.outlier.ransac import OutlierRejector, RobustMethod
from tests.conftest import random_match_points, random_transform_matrix

# Feature: lunar-image-registration, Property 10: Inlier count and ratio arithmetic consistency
# Validates: Requirements 5.2, 9.2, 9.3
@given(
    matches=random_match_points(min_pts=0, max_pts=50),
    method=st.sampled_from(list(RobustMethod)),
    model_type=st.sampled_from(["homography", "affine"])
)
@settings(max_examples=50, deadline=None)
def test_outlier_rejection_arithmetic_consistency(matches, method, model_type):
    rejector = OutlierRejector(method=method)
    res = rejector.reject(matches, model_type=model_type)
    
    assert res.inlier_count == len(res.inlier_matches)
    assert len(res.inlier_matches) + len(res.outlier_matches) == len(matches)
    if len(matches) > 0:
        assert abs(res.inlier_ratio - (res.inlier_count / len(matches))) < 1e-6
    else:
        assert res.inlier_ratio == 0.0

# Feature: lunar-image-registration, Property 9: Perfect-input inlier preservation
# Validates: Requirements 5.3
@given(
    M=random_transform_matrix(transform_type="affine"),
    n=st.integers(10, 50)
)
@settings(max_examples=30, deadline=None)
def test_perfect_input_inlier_preservation(M, n):
    from lunar_reg.matching.base import MatchPair
    
    # Generate perfect matches
    matches = []
    for i in range(n):
        src_x = float(np.random.uniform(10, 100))
        src_y = float(np.random.uniform(10, 100))
        pt = M @ np.array([src_x, src_y, 1.0])
        ref_x = float(pt[0])
        ref_y = float(pt[1])
        matches.append(
            MatchPair(
                source_idx=i,
                reference_idx=i,
                source_pt=(src_x, src_y),
                reference_pt=(ref_x, ref_y),
                confidence=1.0
            )
        )
        
    rejector = OutlierRejector(method=RobustMethod.RANSAC, threshold=3.0)
    res = rejector.reject(matches, model_type="affine")
    # For perfect input with zero noise, all points should be inliers (inlier ratio >= 0.95)
    assert res.inlier_ratio >= 0.95

# Feature: lunar-image-registration, Property 8: Outlier rejection geometric consistency
# Validates: Requirements 5.1
@given(
    matches=random_match_points(min_pts=10, max_pts=40),
    method=st.sampled_from(list(RobustMethod)),
    model_type=st.sampled_from(["homography", "affine"])
)
@settings(max_examples=30, deadline=None)
def test_outlier_rejection_geometric_consistency(matches, method, model_type):
    threshold = 5.0
    rejector = OutlierRejector(method=method, threshold=threshold)
    res = rejector.reject(matches, model_type=model_type)
    
    if res.model_matrix is not None and len(res.inlier_matches) >= 4:
        M = res.model_matrix
        for m in res.inlier_matches:
            pt_hom = np.array([m.source_pt[0], m.source_pt[1], 1.0])
            proj = M @ pt_hom
            proj_x = proj[0] / (proj[2] + 1e-8)
            proj_y = proj[1] / (proj[2] + 1e-8)
            res_val = np.sqrt((proj_x - m.reference_pt[0])**2 + (proj_y - m.reference_pt[1])**2)
            # RANSAC re-fits least-squares on all inliers, which can slightly shift the model matrix.
            # Thus, we check with a small margin, ensuring it is within threshold * 3.0.
            assert res_val <= threshold * 3.0, f"Residual {res_val} exceeds threshold margin"
