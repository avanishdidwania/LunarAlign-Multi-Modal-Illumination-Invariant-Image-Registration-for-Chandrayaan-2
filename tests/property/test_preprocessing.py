from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np
from scipy.stats import pearsonr
import pytest
from lunar_reg.preprocessing.illumination import IlluminationNormalizer
from tests.conftest import random_grayscale_image

# Feature: lunar-image-registration, Property 1: Preprocessing preserves original image data
# Validates: Requirements 2.4
@given(
    image=random_grayscale_image(min_size=64, max_size=128),
    method=st.sampled_from(["phase_congruency", "clahe", "gradient", "lnms"])
)
@settings(max_examples=30, deadline=None)
def test_preprocessing_preserves_original(image, method):
    original_copy = image.copy()
    normalizer = IlluminationNormalizer()
    _ = normalizer.normalize(image, method=method)
    np.testing.assert_array_equal(image, original_copy)

# Feature: lunar-image-registration, Property 2: Illumination invariance of phase congruency
# Validates: Requirements 2.3, 3.1
@given(image=random_grayscale_image(min_size=64, max_size=128))
@settings(max_examples=20, deadline=None)
def test_phase_congruency_illumination_invariance(image):
    normalizer = IlluminationNormalizer()
    
    # Ensure there is some texture
    if np.std(image) < 1.0:
        pytest.skip("Image is too homogeneous for correlation check")
        
    pc_orig = normalizer.phase_congruency(image)
    
    # Create a synthetic illumination gradient
    h, w = image.shape
    x = np.linspace(0.5, 1.5, w)
    y = np.linspace(0.8, 1.2, h)
    xv, yv = np.meshgrid(x, y)
    grad = xv * yv
    
    img_mod = np.clip(image.astype(np.float32) * grad, 0.0, 255.0)
    pc_mod = normalizer.phase_congruency(img_mod)
    
    # Calculate Pearson correlation coefficient
    # Flat arrays are required for pearsonr
    corr, _ = pearsonr(pc_orig.flatten(), pc_mod.flatten())
    
    # If standard deviation is extremely low (flat image), pearsonr can return nan
    if not np.isnan(corr):
        assert corr > 0.8, f"Pearson correlation between phase congruency maps is {corr}, expected > 0.8"
