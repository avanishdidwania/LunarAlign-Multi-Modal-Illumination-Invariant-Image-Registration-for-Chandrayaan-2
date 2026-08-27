import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from lunar_reg.config import (
    RegistrationConfig,
    VALID_ILLUMINATION_METHODS,
    VALID_DETECTION_METHODS,
    VALID_MATCHING_METHODS,
    VALID_OUTLIER_METHODS,
    VALID_INTERPOLATION_METHODS,
    VALID_DEVICES,
    VALID_TRANSFORM_TYPES,
)
from lunar_reg.matching.base import MatchPair

@st.composite
def random_grayscale_image(draw, min_size=64, max_size=256, dtype=np.uint8):
    width = draw(st.integers(min_size, max_size))
    height = draw(st.integers(min_size, max_size))
    if np.issubdtype(dtype, np.integer):
        elements = st.integers(0, 255)
    else:
        elements = st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False)
    
    arr = draw(arrays(dtype=dtype, shape=(height, width), elements=elements))
    return arr

@st.composite
def random_transform_matrix(draw, transform_type=None):
    # Select type if not specified
    if transform_type is None:
        t_type = draw(st.sampled_from(["affine", "projective"]))
    else:
        t_type = transform_type

    scale_x = draw(st.floats(0.5, 2.0))
    scale_y = draw(st.floats(0.5, 2.0))
    theta = draw(st.floats(-np.pi/6, np.pi/6)) # rotation -30 to +30 deg
    tx = draw(st.floats(-50.0, 50.0))
    ty = draw(st.floats(-50.0, 50.0))
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    if t_type == "affine":
        matrix = np.array([
            [scale_x * cos_t, -scale_y * sin_t, tx],
            [scale_x * sin_t, scale_y * cos_t, ty],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
    else:
        px = draw(st.floats(-0.001, 0.001))
        py = draw(st.floats(-0.001, 0.001))
        matrix = np.array([
            [scale_x * cos_t, -scale_y * sin_t, tx],
            [scale_x * sin_t, scale_y * cos_t, ty],
            [px, py, 1.0]
        ], dtype=np.float64)
        
    return matrix

@st.composite
def random_match_points(draw, n=None, min_pts=4, max_pts=50, image_shape=(512, 512)):
    if n is None:
        num_pts = draw(st.integers(min_pts, max_pts))
    else:
        num_pts = n
        
    h, w = image_shape
    matches = []
    for i in range(num_pts):
        src_x = draw(st.floats(0.0, float(w - 1)))
        src_y = draw(st.floats(0.0, float(h - 1)))
        ref_x = draw(st.floats(0.0, float(w - 1)))
        ref_y = draw(st.floats(0.0, float(h - 1)))
        conf = draw(st.floats(0.0, 1.0))
        matches.append(
            MatchPair(
                source_idx=i,
                reference_idx=i,
                source_pt=(src_x, src_y),
                reference_pt=(ref_x, ref_y),
                confidence=conf
            )
        )
    return matches

@st.composite
def random_config(draw):
    illumination_method = draw(st.sampled_from(list(VALID_ILLUMINATION_METHODS)))
    pyramid_levels = draw(st.none() | st.integers(1, 5))
    pyramid_scale_factor = draw(st.floats(0.1, 0.9))
    detection_method = draw(st.sampled_from(list(VALID_DETECTION_METHODS)))
    max_keypoints = draw(st.integers(100, 10000))
    matching_method = draw(st.sampled_from(list(VALID_MATCHING_METHODS)))
    match_threshold = draw(st.floats(0.0, 1.0))
    outlier_method = draw(st.sampled_from(list(VALID_OUTLIER_METHODS)))
    ransac_confidence = draw(st.floats(0.01, 0.9999))
    ransac_max_iters = draw(st.integers(10, 20000))
    transform_type = draw(st.sampled_from(list(VALID_TRANSFORM_TYPES)))
    refine_subpixel = draw(st.booleans())
    refinement_patch_size = draw(st.sampled_from([11, 15, 21, 31]))
    interpolation = draw(st.sampled_from(list(VALID_INTERPOLATION_METHODS)))
    device = draw(st.sampled_from(list(VALID_DEVICES)))
    
    return RegistrationConfig(
        illumination_method=illumination_method,
        pyramid_levels=pyramid_levels,
        pyramid_scale_factor=pyramid_scale_factor,
        detection_method=detection_method,
        max_keypoints=max_keypoints,
        matching_method=matching_method,
        match_threshold=match_threshold,
        outlier_method=outlier_method,
        ransac_confidence=ransac_confidence,
        ransac_max_iters=ransac_max_iters,
        transform_type=transform_type,
        refine_subpixel=refine_subpixel,
        refinement_patch_size=refinement_patch_size,
        interpolation=interpolation,
        device=device
    )

@st.composite
def random_file_extension(draw, exclude_supported=True):
    supported = {".tif", ".tiff", ".img", ".pds", ".jp2", ".png"}
    # Generate random strings of length 1-4 for extension
    ext = draw(st.text(min_size=1, max_size=4, alphabet="abcdefghijklmnopqrstuvwxyz"))
    full_ext = f".{ext}"
    if exclude_supported:
        while full_ext in supported:
            ext = draw(st.text(min_size=1, max_size=4, alphabet="abcdefghijklmnopqrstuvwxyz"))
            full_ext = f".{ext}"
    return full_ext
