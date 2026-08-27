import rasterio
import numpy as np
from typing import Dict, Any

class GeoTiffExporter:
    """
    Geospatial GeoTIFF exporter.
    Saves registered/warped bands while preserving coordinate reference systems (CRS)
    and metadata profiles.
    """

    def export_registered(self, warped_image: np.ndarray,
                          output_path: str,
                          reference_profile: Dict[str, Any]) -> None:
        """
        Export warped image as GeoTIFF using geospatial metadata from reference_profile.
        Supports both single-band (2D) and multi-band (3D) images.
        """
        profile = reference_profile.copy()
        
        # Determine image dimensions and bands
        if warped_image.ndim == 2:
            height, width = warped_image.shape
            count = 1
            bands_data = [warped_image]
        elif warped_image.ndim == 3:
            # Distinguish channel order: (C, H, W) vs (H, W, C)
            if warped_image.shape[0] <= 4:
                count, height, width = warped_image.shape
                bands_data = [warped_image[i] for i in range(count)]
            else:
                height, width, count = warped_image.shape
                bands_data = [warped_image[:, :, i] for i in range(count)]
        else:
            raise ValueError(f"Unsupported warped image dimension: {warped_image.ndim}")
            
        profile.update({
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': count,
            'dtype': str(warped_image.dtype)
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            for i, band in enumerate(bands_data):
                dst.write(band, i + 1)
