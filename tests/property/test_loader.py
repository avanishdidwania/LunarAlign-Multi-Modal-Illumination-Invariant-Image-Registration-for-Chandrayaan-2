import tempfile
from pathlib import Path
from hypothesis import given, settings
import pytest
from lunar_reg.loader.image_loader import ImageLoader, UnsupportedFormatError
from tests.conftest import random_file_extension

# Feature: lunar-image-registration, Property 23: Unsupported format error message content
# Validates: Requirements 1.6
@given(ext=random_file_extension(exclude_supported=True))
@settings(max_examples=50, deadline=None)
def test_unsupported_format_error_message(ext):
    loader = ImageLoader()
    
    # Create a temporary file with the unsupported extension
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        
    try:
        with pytest.raises(UnsupportedFormatError) as exc_info:
            loader.load(tmp_path)
            
        err_msg = str(exc_info.value)
        # Check that the invalid extension is in the error message
        assert ext in err_msg.lower()
        # Check that all supported formats are listed in the error message
        for supported_ext in loader.SUPPORTED_FORMATS:
            assert supported_ext in err_msg.lower()
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
