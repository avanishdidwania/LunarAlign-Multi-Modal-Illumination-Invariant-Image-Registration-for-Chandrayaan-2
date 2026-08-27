from hypothesis import given, settings
from hypothesis import strategies as st
from pathlib import Path
import pytest
from lunar_reg.config import RegistrationConfig
from tests.conftest import random_config

# Feature: lunar-image-registration, Property 22: Configuration validation
# Validates: Requirements 13.1
@given(config=random_config())
@settings(max_examples=100)
def test_config_validation_valid(config):
    # Calling validate() should not raise any exceptions
    config.validate()

@given(
    method=st.text(min_size=1, max_size=15).filter(
        lambda x: x not in {"phase_congruency", "clahe", "gradient", "lnms"}
    )
)
@settings(max_examples=20)
def test_config_validation_invalid_illumination(method):
    with pytest.raises(ValueError):
        RegistrationConfig(illumination_method=method)

@given(
    method=st.text(min_size=1, max_size=15).filter(
        lambda x: x not in {"sift", "superpoint"}
    )
)
@settings(max_examples=20)
def test_config_validation_invalid_detection(method):
    with pytest.raises(ValueError):
        RegistrationConfig(detection_method=method)

@given(
    method=st.text(min_size=1, max_size=15).filter(
        lambda x: x not in {"bf", "lightglue", "loftr"}
    )
)
@settings(max_examples=20)
def test_config_validation_invalid_matching(method):
    with pytest.raises(ValueError):
        RegistrationConfig(matching_method=method)

@given(
    method=st.text(min_size=1, max_size=15).filter(
        lambda x: x not in {"ransac", "magsac++", "lmeds"}
    )
)
@settings(max_examples=20)
def test_config_validation_invalid_outlier(method):
    with pytest.raises(ValueError):
        RegistrationConfig(outlier_method=method)

@given(
    t_type=st.text(min_size=1, max_size=15).filter(
        lambda x: x not in {"affine", "projective"}
    )
)
@settings(max_examples=20)
def test_config_validation_invalid_transform(t_type):
    with pytest.raises(ValueError):
        RegistrationConfig(transform_type=t_type)

@given(
    device=st.text(min_size=1, max_size=15).filter(
        lambda x: x not in {"auto", "cuda", "cpu"}
    )
)
@settings(max_examples=20)
def test_config_validation_invalid_device(device):
    with pytest.raises(ValueError):
        RegistrationConfig(device=device)

# Feature: lunar-image-registration, Property 21: Pipeline end-to-end exception containment
# Validates: Requirements 12.1
@given(
    config=random_config(),
    src_path=st.text(min_size=1, max_size=50).map(Path),
    ref_path=st.text(min_size=1, max_size=50).map(Path)
)
@settings(max_examples=10, deadline=None)
def test_pipeline_exception_containment(config, src_path, ref_path):
    from lunar_reg.pipeline import RegistrationPipeline
    pipeline = RegistrationPipeline(config)
    
    # Run pipeline with completely invalid paths
    # It must return a failed RegistrationResult without raising any exception
    res = pipeline.run(src_path, ref_path, Path("./output_dummy"))
    
    assert res.success is False
    assert res.error_message is not None
