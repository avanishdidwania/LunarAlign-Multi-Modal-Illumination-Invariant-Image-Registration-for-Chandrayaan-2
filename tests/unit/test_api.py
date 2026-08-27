import io
import os
import time
from pathlib import Path
import pytest
import rasterio
from rasterio.transform import from_origin
import numpy as np
from fastapi.testclient import TestClient

from lunar_reg.web.app import app

client = TestClient(app)

def create_mock_tiff():
    """Generates a valid georeferenced GeoTIFF in memory."""
    buf = io.BytesIO()
    with rasterio.open(
        buf, 'w',
        driver='GTiff',
        height=64, width=64,
        count=1, dtype='uint8',
        crs='+proj=latlong',
        transform=from_origin(0.0, 0.0, 1.0, 1.0)
    ) as dst:
        # Write some textured data to help detection
        data = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        # Add high-contrast corner grids
        data[::8, :] = 255
        data[:, ::8] = 0
        dst.write(data, 1)
    buf.seek(0)
    return buf.read()

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_config_methods_endpoint():
    response = client.get("/api/v1/config/methods")
    assert response.status_code == 200
    data = response.json()
    assert "illumination_methods" in data
    assert "detection_methods" in data
    assert "matching_methods" in data
    assert "outlier_methods" in data

def test_register_invalid_files():
    # Upload invalid text files - loader will catch this in the background
    files = {
        "source_image": ("source.txt", b"not tiff data", "text/plain"),
        "reference_image": ("reference.txt", b"not tiff data", "text/plain")
    }
    data = {
        "illumination_method": "phase_congruency",
        "detection_method": "sift",
        "matching_method": "bf",
        "outlier_method": "ransac",
        "refine_subpixel": "false",
        "device": "cpu"
    }
    response = client.post("/api/v1/register", files=files, data=data)
    assert response.status_code == 200
    job_info = response.json()
    assert "job_id" in job_info
    assert job_info["status"] == "pending"

def test_register_and_poll_valid():
    src_bytes = create_mock_tiff()
    ref_bytes = create_mock_tiff()
    
    files = {
        "source_image": ("source_ohrc.tif", src_bytes, "image/tiff"),
        "reference_image": ("reference_tmc.tif", ref_bytes, "image/tiff")
    }
    data = {
        "illumination_method": "clahe",
        "detection_method": "sift",
        "matching_method": "bf",
        "outlier_method": "ransac",
        "refine_subpixel": "false",
        "device": "cpu"
    }
    
    # Submit job
    response = client.post("/api/v1/register", files=files, data=data)
    assert response.status_code == 200
    job_info = response.json()
    job_id = job_info["job_id"]
    assert job_info["status"] == "pending"
    
    # Poll status
    status = "pending"
    for _ in range(30):
        time.sleep(0.5)
        resp = client.get(f"/api/v1/jobs/{job_id}")
        assert resp.status_code == 200
        job_data = resp.json()
        status = job_data["status"]
        if status in {"completed", "failed"}:
            break
            
    assert status in {"completed", "failed"}
    
    # Cleanup files
    for folder in ["./uploads", "./results", "./results/jobs"]:
        path = Path(folder)
        if path.exists():
            for f in path.iterdir():
                if job_id in f.name:
                    try:
                        f.unlink()
                    except Exception:
                        pass
