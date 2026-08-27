import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
import pytest
from lunar_reg.warping.warper import ImageWarper
from tests.conftest import random_grayscale_image

# Feature: lunar-image-registration, Property 15: Warped image spatial preservation
# Validates: Requirements 8.1
@given(
    image=random_grayscale_image(min_size=64, max_size=128),
    interpolation=st.sampled_from(["nearest", "bilinear", "bicubic"])
)
@settings(max_examples=20, deadline=None)
def test_warping_spatial_preservation(image, interpolation):
    warper = ImageWarper()
    identity = np.eye(3, dtype=np.float64)
    
    warped = warper.warp(image, identity, image.shape, interpolation=interpolation)
    
    assert warped.shape == image.shape
    np.testing.assert_array_equal(warped, image)

# Feature: lunar-image-registration, Property 16: Warped border conservation
# Validates: Requirements 8.3
@given(
    image=random_grayscale_image(min_size=64, max_size=128),
    tx=st.floats(10.0, 30.0),
    ty=st.floats(10.0, 30.0),
    border_val=st.floats(0.0, 255.0)
)
@settings(max_examples=20, deadline=None)
def test_warping_border_conservation(image, tx, ty, border_val):
    warper = ImageWarper()
    
    # Translation matrix: maps (x, y) to (x + tx, y + ty)
    T = np.array([
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    warped = warper.warp(image, T, image.shape, interpolation="nearest", border_value=border_val)
    
    # Destination pixels that correspond to coordinates outside the source image (x < 0, y < 0)
    # should be filled with border_val (cast to the image's data type by OpenCV).
    int_tx = int(np.floor(tx))
    int_ty = int(np.floor(ty))
    
    if int_ty > 0:
        assert np.all(np.abs(warped[0:int_ty, :].astype(np.float32) - border_val) <= 1.0)
    if int_tx > 0:
        assert np.all(np.abs(warped[:, 0:int_tx].astype(np.float32) - border_val) <= 1.0)
