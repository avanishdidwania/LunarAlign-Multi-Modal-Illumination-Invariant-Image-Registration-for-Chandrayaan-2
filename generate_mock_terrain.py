"""
Generate overlapping, high-texture mock lunar images (TMC and LRO NAC simulated)
to run a successful registration demo in the portal.
"""
import rasterio
from rasterio.transform import from_origin
import numpy as np
import scipy.ndimage


def generate_mock_crater_terrain(size=512):
    # Start with random noise
    np.random.seed(42)
    terrain = np.random.normal(128, 20, (size, size))
    
    # Generate some mock craters (circular structures)
    for _ in range(15):
        cx, cy = np.random.randint(50, size - 50, 2)
        r = np.random.randint(15, 60)
        depth = np.random.randint(40, 100)
        
        # Create distance mask
        y, x = np.ogrid[-cy:size-cy, -cx:size-cx]
        mask = x*x + y*y <= r*r
        
        # Draw crater rim and basin
        terrain[mask] = np.clip(terrain[mask] - depth * (1 - np.sqrt(x*x + y*y)[mask]/r), 0, 255)
        
        # Draw crater rim
        rim_mask = (x*x + y*y >= (r-3)**2) & (x*x + y*y <= (r+2)**2)
        terrain[rim_mask] = np.clip(terrain[rim_mask] + depth * 0.4, 0, 255)
        
    # Smooth slightly to simulate camera lens
    terrain = scipy.ndimage.gaussian_filter(terrain, sigma=1.5)
    return np.clip(terrain, 0, 255).astype(np.uint8)


def main():
    size = 512
    # Generate underlying lunar-like terrain
    base_terrain = generate_mock_crater_terrain(size)
    
    # 1. Source Image (simulated TMC-2, rotated and shifted slightly)
    # Rotate by 5 degrees and shift by 10 pixels
    source_img = scipy.ndimage.rotate(base_terrain, 5.0, reshape=False, mode='nearest')
    source_img = scipy.ndimage.shift(source_img, (10, -5), mode='nearest')
    
    # 2. Reference Image (simulated LRO NAC, original terrain with different illumination/contrast)
    # Modify contrast and brightness to simulate different sensor/sun angle
    ref_img = np.clip((base_terrain.astype(float) - 128) * 1.3 + 110, 0, 255).astype(np.uint8)
    
    # Write Source GeoTIFF
    with rasterio.open(
        'mock_source.tif', 'w',
        driver='GTiff',
        height=size, width=size,
        count=1, dtype='uint8',
        crs='EPSG:4326',
        transform=from_origin(32.5, -69.5, 0.0001, 0.0001)
    ) as dst:
        dst.write(source_img, 1)
        
    # Write Reference GeoTIFF (overlapping spatial bounds)
    with rasterio.open(
        'mock_reference.tif', 'w',
        driver='GTiff',
        height=size, width=size,
        count=1, dtype='uint8',
        crs='EPSG:4326',
        transform=from_origin(32.5005, -69.5005, 0.0001, 0.0001)
    ) as dst:
        dst.write(ref_img, 1)
        
    print("SUCCESS: Generated overlapping mock terrain images:")
    print("  - mock_source.tif")
    print("  - mock_reference.tif")


if __name__ == "__main__":
    main()
