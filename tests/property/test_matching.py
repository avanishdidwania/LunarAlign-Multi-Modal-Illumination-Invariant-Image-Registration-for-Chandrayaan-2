import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import pytest
from lunar_reg.detection.base import DetectionResult, Keypoint
from lunar_reg.matching.bf_matcher import BFMatcher
from lunar_reg.matching.lightglue_matcher import LightGlueMatcher
from tests.conftest import random_grayscale_image

@st.composite
def random_detection_result(draw, image_shape=(128, 128), descriptor_dim=128):
    num_kps = draw(st.integers(0, 50))
    keypoints = []
    for i in range(num_kps):
        keypoints.append(
            Keypoint(
                x=draw(st.floats(0.0, float(image_shape[1]-1))),
                y=draw(st.floats(0.0, float(image_shape[0]-1))),
                scale=draw(st.floats(1.0, 10.0)),
                orientation=draw(st.floats(0.0, 2.0*np.pi)),
                response=draw(st.floats(0.0, 1.0))
            )
        )
    # Generate descriptors
    descs = draw(arrays(
        dtype=np.float32,
        shape=(num_kps, descriptor_dim),
        elements=st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False)
    ))
    # Normalize descriptors to unit length
    if num_kps > 0:
        norm = np.linalg.norm(descs, axis=1, keepdims=True) + 1e-8
        descs = descs / norm
    return DetectionResult(keypoints=keypoints, descriptors=descs, image_shape=image_shape)

# Feature: lunar-image-registration, Property 7: Match output validity
# Validates: Requirements 4.1, 4.3
@given(
    src_res=random_detection_result(descriptor_dim=128),
    ref_res=random_detection_result(descriptor_dim=128),
    ratio=st.floats(0.5, 0.95)
)
@settings(max_examples=30, deadline=None)
def test_bf_matcher_output_validity(src_res, ref_res, ratio):
    matcher = BFMatcher(ratio_threshold=ratio)
    result = matcher.match(src_res, ref_res)
    
    for match in result.matches:
        # Check source index bounds
        assert 0 <= match.source_idx < len(src_res.keypoints)
        # Check reference index bounds
        assert 0 <= match.reference_idx < len(ref_res.keypoints)
        # Check confidence range [0, 1]
        assert 0.0 <= match.confidence <= 1.0
        # Check coordinate values match keypoint coordinates
        src_kp = src_res.keypoints[match.source_idx]
        ref_kp = ref_res.keypoints[match.reference_idx]
        assert match.source_pt == (src_kp.x, src_kp.y)
        assert match.reference_pt == (ref_kp.x, ref_kp.y)

# Feature: lunar-image-registration, Property 7: Match output validity (LightGlue)
# Validates: Requirements 4.1, 4.3
@given(
    src_res=random_detection_result(descriptor_dim=256),
    ref_res=random_detection_result(descriptor_dim=256),
    threshold=st.floats(0.0, 0.8)
)
@settings(max_examples=10, deadline=None)
def test_lightglue_matcher_output_validity(src_res, ref_res, threshold):
    try:
        matcher = LightGlueMatcher(match_threshold=threshold)
    except Exception as e:
        pytest.skip(f"LightGlue matcher failed to load: {e}")
        
    result = matcher.match(src_res, ref_res)
    
    for match in result.matches:
        assert 0 <= match.source_idx < len(src_res.keypoints)
        assert 0 <= match.reference_idx < len(ref_res.keypoints)
        assert 0.0 <= match.confidence <= 1.0
        src_kp = src_res.keypoints[match.source_idx]
        ref_kp = ref_res.keypoints[match.reference_idx]
        assert match.source_pt == (src_kp.x, src_kp.y)
        assert match.reference_pt == (ref_kp.x, ref_kp.y)
