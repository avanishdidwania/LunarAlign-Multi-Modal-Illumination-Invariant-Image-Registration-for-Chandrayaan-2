import warnings

import numpy as np
import pytest

from lunar_reg.preprocessing.illumination import IlluminationNormalizer


def _assert_no_overflow_warning(func):
    """Run func and assert no overflow RuntimeWarning is raised."""
    with warnings.catch_warnings():
        # Promote any RuntimeWarning (incl. overflow) to an error.
        warnings.simplefilter("error", RuntimeWarning)
        # np.errstate also converts numpy floating/integer errors to warnings/errors.
        with np.errstate(over="raise"):
            return func()


def test_clahe_int16_large_dynamic_range_no_overflow():
    """int16 image spanning -30000..+30000 (range 60000 > int16 max 32767).

    The old code computed ``image - img_min`` in native int16, overflowing and
    emitting a RuntimeWarning while corrupting the output. The fix casts to
    float32 first.
    """
    rng = np.random.default_rng(0)
    img = rng.integers(-30000, 30001, size=(256, 256)).astype(np.int16)
    # Guarantee the extreme min/max are present to force the overflow condition.
    img[0, 0] = -30000
    img[0, 1] = 30000
    assert img.min() == -30000
    assert img.max() == 30000

    out = _assert_no_overflow_warning(lambda: IlluminationNormalizer().clahe(img))

    assert out.dtype == np.uint8
    assert out.shape == img.shape
    assert out.min() >= 0
    assert out.max() <= 255


def test_clahe_uint16_large_dynamic_range_no_overflow():
    """uint16 image spanning 0..60000 (range 60000 > int16 max 32767)."""
    rng = np.random.default_rng(1)
    img = rng.integers(0, 60001, size=(256, 256)).astype(np.uint16)
    img[0, 0] = 0
    img[0, 1] = 60000
    assert img.min() == 0
    assert img.max() == 60000

    out = _assert_no_overflow_warning(lambda: IlluminationNormalizer().clahe(img))

    assert out.dtype == np.uint8
    assert out.shape == img.shape
    assert out.min() >= 0
    assert out.max() <= 255


def test_clahe_constant_image_no_overflow():
    """A flat image (img_max == img_min) takes the zeros_like branch safely.

    CLAHE may still emit a constant non-zero value, so we only assert the
    branch runs without overflow and returns a valid, uniform uint8 array.
    """
    img = np.full((64, 64), 12345, dtype=np.int16)
    out = _assert_no_overflow_warning(lambda: IlluminationNormalizer().clahe(img))
    assert out.dtype == np.uint8
    assert out.shape == img.shape
    # A flat input yields a flat output.
    assert out.min() == out.max()


def test_clahe_uint8_passthrough():
    """uint8 input must still be processed directly without a normalization step."""
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=(64, 64)).astype(np.uint8)
    out = IlluminationNormalizer().clahe(img)
    assert out.dtype == np.uint8
    assert out.shape == img.shape
