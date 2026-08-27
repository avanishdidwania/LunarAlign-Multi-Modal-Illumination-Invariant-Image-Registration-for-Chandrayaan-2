from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np
from lunar_reg.preprocessing.pyramid import PyramidBuilder
from tests.conftest import random_grayscale_image

# Feature: lunar-image-registration, Property 3: Multi-scale pyramid structure invariants
# Validates: Requirements 2.2
@given(
    image=random_grayscale_image(min_size=64, max_size=256),
    n_levels=st.integers(1, 5),
    scale_factor=st.floats(0.1, 0.9)
)
@settings(max_examples=50, deadline=None)
def test_pyramid_structure_invariants(image, n_levels, scale_factor):
    builder = PyramidBuilder()
    pyramid = builder.build(image, n_levels=n_levels, scale_factor=scale_factor)
    
    # (a) Exactly the specified number of levels
    assert len(pyramid.levels) == n_levels
    assert len(pyramid.scale_factors) == n_levels
    
    # (c) All levels with the same dtype as input
    for lvl in pyramid.levels:
        assert lvl.dtype == image.dtype
        
    # (b) Dimension progression scaling check
    for i in range(1, n_levels):
        prev_h, prev_w = pyramid.levels[i-1].shape[:2]
        curr_h, curr_w = pyramid.levels[i].shape[:2]
        
        expected_w = max(1, int(round(prev_w * scale_factor)))
        expected_h = max(1, int(round(prev_h * scale_factor)))
        
        assert abs(curr_w - expected_w) <= 1
        assert abs(curr_h - expected_h) <= 1
