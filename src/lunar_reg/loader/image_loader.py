import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import rasterio

class SensorType(Enum):
    OHRC = "ohrc"            # 0.25m panchromatic
    TMC2 = "tmc2"            # 5m panchromatic triplet
    IIRS = "iirs"            # Spectral (hyperspectral)
    LRO_NAC = "lro_nac"      # ~0.5m panchromatic
    SELENE = "selene"        # ~10m panchromatic
    UNKNOWN = "unknown"

DEFAULT_RESOLUTIONS = {
    SensorType.OHRC: 0.25,
    SensorType.LRO_NAC: 0.5,
    SensorType.TMC2: 5.0,
    SensorType.SELENE: 10.0,
    SensorType.IIRS: 80.0,
    SensorType.UNKNOWN: 1.0,
}

@dataclass
class ImageMetadata:
    file_path: Path
    sensor_type: SensorType
    spatial_resolution: float          # meters/pixel
    width: int
    height: int
    num_bands: int
    crs: Optional[str]                    # Coordinate reference system (WKT or PROJ)
    geotransform: Optional[Tuple[float, ...]]  # GDAL-style affine geotransform
    sun_elevation: Optional[float]        # degrees
    sun_azimuth: Optional[float]          # degrees
    acquisition_time: Optional[str]
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LoadedImage:
    data: np.ndarray                   # Shape: (H, W) for panchromatic, (H, W, B) for spectral
    metadata: ImageMetadata

class UnsupportedFormatError(Exception):
    """Exception raised when an unsupported file format is loaded."""
    pass

class CorruptFileError(Exception):
    """Exception raised when a file is corrupted or unreadable."""
    pass

class ImageLoader:
    """Loads satellite imagery using GDAL/Rasterio with format auto-detection."""

    SUPPORTED_FORMATS: List[str] = [".tif", ".tiff", ".img", ".pds", ".jp2", ".png"]

    def validate_format(self, file_path: Path) -> bool:
        """Check if file format is supported."""
        ext = file_path.suffix.lower()
        return ext in self.SUPPORTED_FORMATS

    def detect_sensor_type(self, file_path: Path, metadata: Dict[str, Any]) -> SensorType:
        """Infer sensor type from file metadata or filename conventions."""
        name = file_path.name.lower()
        if "ohrc" in name:
            return SensorType.OHRC
        elif "tmc" in name:
            return SensorType.TMC2
        elif "iirs" in name:
            return SensorType.IIRS
        elif "nac" in name or "lro" in name:
            return SensorType.LRO_NAC
        elif "selene" in name or "kaguya" in name:
            return SensorType.SELENE
        
        # Check metadata values
        sensor_meta = metadata.get("SENSOR_NAME", "").lower() or metadata.get("INSTRUMENT_NAME", "").lower()
        if "ohrc" in sensor_meta:
            return SensorType.OHRC
        elif "tmc" in sensor_meta:
            return SensorType.TMC2
        elif "iirs" in sensor_meta:
            return SensorType.IIRS
        elif "nac" in sensor_meta or "lro" in sensor_meta:
            return SensorType.LRO_NAC
        elif "selene" in sensor_meta or "kaguya" in sensor_meta:
            return SensorType.SELENE
            
        return SensorType.UNKNOWN

    def load(self, file_path: Path) -> LoadedImage:
        """Load image and extract metadata. Raises UnsupportedFormatError or CorruptFileError."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if not self.validate_format(file_path):
            raise UnsupportedFormatError(
                f"Unsupported format '{file_path.suffix}'. Supported formats are: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        try:
            with rasterio.open(file_path) as src:
                # If image is extremely large (e.g. orbit tracks), use windowed reading
                # to only load a central 4096x4096 patch. This avoids OOMs and takes < 1 second.
                MAX_DIM = 4096
                if src.height > MAX_DIM or src.width > MAX_DIM:
                    new_h = min(src.height, MAX_DIM)
                    new_w = min(src.width, MAX_DIM)
                    dy = (src.height - new_h) // 2
                    dx = (src.width - new_w) // 2
                    from rasterio.windows import Window
                    win = Window(dx, dy, new_w, new_h)
                    data = src.read(window=win)
                else:
                    data = src.read()
                
                # Check for empty or invalid data
                if data.size == 0:
                    raise CorruptFileError(f"Image has empty data array: {file_path}")
                
                # Reshape data to (H, W) or (H, W, B)
                if data.shape[0] == 1:
                    data = data[0]  # (H, W)
                else:
                    data = np.transpose(data, (1, 2, 0))  # (H, W, B)
                
                # Extract georeferencing
                crs = src.crs.to_wkt() if src.crs else None
                t = src.transform
                geotransform = (t.c, t.a, t.b, t.f, t.d, t.e) if t else None
                
                # Extra metadata
                tags = src.tags()
                
                sensor_type = self.detect_sensor_type(file_path, tags)
                
                # Spatial resolution
                if t:
                    spatial_resolution = abs(t.a)
                else:
                    spatial_resolution = DEFAULT_RESOLUTIONS.get(sensor_type, 1.0)
                
                # Extract sun angles if available
                sun_elevation = None
                sun_azimuth = None
                for key in ["SUN_ELEVATION", "SOLAR_ELEVATION", "elevation", "Elevation"]:
                    if key in tags:
                        try:
                            sun_elevation = float(tags[key])
                            break
                        except ValueError:
                            pass
                for key in ["SUN_AZIMUTH", "SOLAR_AZIMUTH", "azimuth", "Azimuth"]:
                    if key in tags:
                        try:
                            sun_azimuth = float(tags[key])
                            break
                        except ValueError:
                            pass
                            
                acquisition_time = tags.get("ACQUISITION_TIME") or tags.get("START_TIME") or tags.get("time")

                metadata = ImageMetadata(
                    file_path=file_path,
                    sensor_type=sensor_type,
                    spatial_resolution=spatial_resolution,
                    width=src.width,
                    height=src.height,
                    num_bands=src.count,
                    crs=crs,
                    geotransform=geotransform,
                    sun_elevation=sun_elevation,
                    sun_azimuth=sun_azimuth,
                    acquisition_time=acquisition_time,
                    extra=tags
                )
                
                return LoadedImage(data=data, metadata=metadata)

        except rasterio.errors.RasterioIOError as e:
            raise CorruptFileError(f"Failed to open image file {file_path}: {str(e)}") from e
        except Exception as e:
            if isinstance(e, (UnsupportedFormatError, CorruptFileError, FileNotFoundError)):
                raise e
            raise CorruptFileError(f"Error loading image file {file_path}: {str(e)}") from e
