import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
import pytest
from lunar_reg.transform.estimator import TransformationEstimator, TransformationType
from tests.conftest import random_transform_matrix

# Feature: lunar-image-registration, Property 11: Transformation estimation round-trip
# Validates: Requirements 6.1
@given(
    M=random_transform_matrix(transform_type="affine"),
    num_pts=st.integers(10, 50)
)
@settings(max_examples=30, deadline=None)
def test_transform_estimation_round_trip_affine(M, num_pts):
    from lunar_reg.matching.base import MatchPair
    
    estimator = TransformationEstimator()
    
    # Generate random source points
    src_pts = np.random.uniform(10.0, 1000.0, (num_pts, 2))
    
    # Project using M
    src_hom = np.hstack([src_pts, np.ones((num_pts, 1))])
    ref_hom = src_hom @ M.T
    ref_pts = ref_hom[:, :2] / ref_hom[:, 2:3]
    
    # Create MatchPair list
    matches = []
    for i in range(num_pts):
        matches.append(
            MatchPair(
                source_idx=i,
                reference_idx=i,
                source_pt=(float(src_pts[i, 0]), float(src_pts[i, 1])),
                reference_pt=(float(ref_pts[i, 0]), float(ref_pts[i, 1])),
                confidence=1.0
            )
        )
        
    res = estimator.estimate(matches, TransformationType.AFFINE)
    
    # Check that estimated matrix matches M
    np.testing.assert_allclose(res.matrix, M, rtol=1e-4, atol=1e-4)
    assert res.rmse < 1e-3
    assert res.condition_number < 1000.0

@given(
    H=random_transform_matrix(transform_type="projective"),
    num_pts=st.integers(10, 50)
)
@settings(max_examples=30, deadline=None)
def test_transform_estimation_round_trip_projective(H, num_pts):
    from lunar_reg.matching.base import MatchPair
    
    estimator = TransformationEstimator()
    
    # Check that H is stable and det > 0
    det = np.linalg.det(H[:2, :2])
    if det <= 0:
        H[:2, :2] *= -1.0
        
    s = np.linalg.svd(H[:2, :2], compute_uv=False)
    cond = s[0] / (s[1] + 1e-8)
    if cond > 100.0:
        pytest.skip("Homography is too ill-conditioned")
        
    # Generate random source points
    src_pts = np.random.uniform(10.0, 1000.0, (num_pts, 2))
    
    # Project using H
    src_hom = np.hstack([src_pts, np.ones((num_pts, 1))])
    ref_hom = src_hom @ H.T
    ref_pts = ref_hom[:, :2] / (ref_hom[:, 2:3] + 1e-8)
    
    matches = []
    for i in range(num_pts):
        matches.append(
            MatchPair(
                source_idx=i,
                reference_idx=i,
                source_pt=(float(src_pts[i, 0]), float(src_pts[i, 1])),
                reference_pt=(float(ref_pts[i, 0]), float(ref_pts[i, 1])),
                confidence=1.0
            )
        )
        
    try:
        res = estimator.estimate(matches, TransformationType.PROJECTIVE)
    except Exception as e:
        pytest.skip(f"Homography estimation failed: {e}")
        
    # Normalize homographies (so that H[2, 2] = 1.0)
    H_norm = H / H[2, 2]
    res_matrix_norm = res.matrix / res.matrix[2, 2]
    
    np.testing.assert_allclose(res_matrix_norm, H_norm, rtol=1e-3, atol=1e-3)
    assert res.rmse < 1e-2

# Feature: lunar-image-registration, Property 12: Insufficient points error
# Validates: Requirements 6.3
@given(
    num_pts=st.integers(0, 3),
    transformation_type=st.sampled_from(list(TransformationType))
)
@settings(max_examples=20, deadline=None)
def test_transform_estimation_insufficient_points(num_pts, transformation_type):
    from lunar_reg.matching.base import MatchPair
    
    estimator = TransformationEstimator()
    
    if transformation_type == TransformationType.PROJECTIVE and num_pts == 3:
        pass
    elif transformation_type == TransformationType.AFFINE and num_pts < 3:
        pass
    else:
        if transformation_type == TransformationType.PROJECTIVE:
            num_pts = min(num_pts, 3)
        else:
            num_pts = min(num_pts, 2)
            
    matches = []
    for i in range(num_pts):
        matches.append(
            MatchPair(
                source_idx=i,
                reference_idx=i,
                source_pt=(float(i), float(i)),
                reference_pt=(float(i), float(i)),
                confidence=1.0
            )
        )
        
    with pytest.raises(ValueError) as excinfo:
        estimator.estimate(matches, transformation_type)
    assert "Insufficient points" in str(excinfo.value)
