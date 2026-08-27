import tempfile
import os
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
import pytest
from lunar_reg.exporter.geotiff import GeoTiffExporter
from tests.conftest import random_grayscale_image

# Feature: lunar-image-registration, Property 19: Export spatial metadata preservation
# Feature: lunar-image-registration, Property 20: Export pixel preservation
# Validates: Requirements 10.1, 10.2, 10.3
@given(
    image=random_grayscale_image(min_size=64, max_size=128),
    west=st.floats(-180.0, 180.0),
    north=st.floats(-90.0, 90.0),
    res_x=st.floats(0.1, 10.0),
    res_y=st.floats(0.1, 10.0)
)
@settings(max_examples=15, deadline=None)
def test_geotiff_export_preservation(image, west, north, res_x, res_y):
    h, w = image.shape
    
    transform = from_origin(west, north, res_x, res_y)
    crs = CRS.from_epsg(4326)
    
    ref_profile = {
        'driver': 'GTiff',
        'height': h,
        'width': w,
        'count': 1,
        'dtype': str(image.dtype),
        'crs': crs,
        'transform': transform,
        'nodata': 0
    }
    
    exporter = GeoTiffExporter()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "exported.tif")
        exporter.export_registered(image, out_path, ref_profile)
        
        # Read back and verify
        with rasterio.open(out_path) as src:
            # Property 19: Spatial metadata check
            assert src.crs == ref_profile['crs']
            np.testing.assert_allclose(src.transform, ref_profile['transform'], atol=1e-5)
            assert src.width == w
            assert src.height == h
            assert src.count == 1
            
            # Property 20: Pixel data check
            read_img = src.read(1)
            np.testing.assert_array_equal(read_img, image)
