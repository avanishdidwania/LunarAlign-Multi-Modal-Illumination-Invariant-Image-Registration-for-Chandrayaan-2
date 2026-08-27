import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from pathlib import Path
from lunar_reg.loader.image_loader import ImageLoader, SensorType, UnsupportedFormatError, CorruptFileError

def create_mock_raster(path: Path, driver: str = "GTiff", width: int = 10, height: int = 10, crs: str = "EPSG:4326", tags: dict = None):
    data = np.zeros((1, height, width), dtype=np.uint8)
    transform = from_origin(0, 0, 1.0, 1.0)
    with rasterio.open(
        path,
        "w",
        driver=driver,
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(data)
        if tags:
            dst.update_tags(**tags)

def test_load_ohrc(tmp_path):
    loader = ImageLoader()
    ohrc_path = tmp_path / "test_ohrc_image.tif"
    tags = {
        "SENSOR_NAME": "OHRC",
        "SUN_ELEVATION": "32.5",
        "SUN_AZIMUTH": "145.2",
        "ACQUISITION_TIME": "2024-03-12T04:22:15Z"
    }
    create_mock_raster(ohrc_path, tags=tags)
    
    loaded = loader.load(ohrc_path)
    assert loaded.metadata.sensor_type == SensorType.OHRC
    assert loaded.metadata.spatial_resolution == 1.0  # From transform pixel size (1.0)
    assert loaded.metadata.width == 10
    assert loaded.metadata.height == 10
    assert loaded.metadata.num_bands == 1
    assert loaded.metadata.sun_elevation == 32.5
    assert loaded.metadata.sun_azimuth == 145.2
    assert loaded.metadata.acquisition_time == "2024-03-12T04:22:15Z"
    assert loaded.data.shape == (10, 10)

def test_load_tmc2(tmp_path):
    loader = ImageLoader()
    tmc_path = tmp_path / "test_tmc_image.tif"
    tags = {
        "INSTRUMENT_NAME": "TMC-2",
        "SOLAR_ELEVATION": "15.0"
    }
    create_mock_raster(tmc_path, tags=tags)
    
    loaded = loader.load(tmc_path)
    assert loaded.metadata.sensor_type == SensorType.TMC2
    assert loaded.metadata.sun_elevation == 15.0

def test_load_lro_nac(tmp_path):
    loader = ImageLoader()
    lro_path = tmp_path / "nac_lro_image.tiff"
    create_mock_raster(lro_path)
    
    loaded = loader.load(lro_path)
    assert loaded.metadata.sensor_type == SensorType.LRO_NAC

def test_load_selene(tmp_path):
    loader = ImageLoader()
    # SELENE usually detected by filename
    selene_path = tmp_path / "selene_image.png"
    create_mock_raster(selene_path, driver="PNG")
    
    loaded = loader.load(selene_path)
    assert loaded.metadata.sensor_type == SensorType.SELENE

def test_load_unsupported_format(tmp_path):
    loader = ImageLoader()
    txt_path = tmp_path / "test.txt"
    txt_path.write_text("not a raster image")
    
    with pytest.raises(UnsupportedFormatError):
        loader.load(txt_path)

def test_load_missing_file():
    loader = ImageLoader()
    with pytest.raises(FileNotFoundError):
        loader.load(Path("nonexistent_file.tif"))

def test_load_corrupt_file(tmp_path):
    loader = ImageLoader()
    corrupt_path = tmp_path / "corrupt.tif"
    corrupt_path.write_text("corrupted content that is not tiff format")
    
    with pytest.raises(CorruptFileError):
        loader.load(corrupt_path)
