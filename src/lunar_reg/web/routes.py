import os
import uuid
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import rasterio

from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException, Form, Request
from fastapi.responses import FileResponse, JSONResponse

from lunar_reg.config import RegistrationConfig
from lunar_reg.pipeline import RegistrationPipeline
from lunar_reg.web.schemas import JobResponse, RegistrationResultSchema, MatchPointSchema, QualityMetricsSchema

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

# Configure directories
UPLOADS_DIR = Path("./uploads")
RESULTS_DIR = Path("./results")
JOBS_DIR = RESULTS_DIR / "jobs"

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

def run_pipeline_job(
    job_id: str,
    src_path: Path,
    ref_path: Path,
    output_dir: Path,
    config: RegistrationConfig
):
    """Background task executor for image registration."""
    job_file = JOBS_DIR / f"{job_id}.json"
    
    # Update status to running
    job_data = {
        "job_id": job_id,
        "status": "running",
        "result": None,
        "registered_image_url": None,
        "source_path": str(src_path),
        "reference_path": str(ref_path)
    }
    with open(job_file, "w") as f:
        json.dump(job_data, f)
        
    try:
        pipeline = RegistrationPipeline(config)
        res = pipeline.run(src_path, ref_path, output_dir)
        
        if res.success:
            match_points = []
            for rm in res.refined_matches:
                match_points.append({
                    "source_pt": list(rm.source_pt),
                    "reference_pt": list(rm.reference_pt),
                    "confidence": float(rm.ncc_score)
                })
                
            q_metrics = None
            if res.quality_metrics is not None:
                qm = res.quality_metrics
                q_metrics = {
                    "ssim": float(qm.ssim),
                    "psnr": float(qm.psnr),
                    "mutual_information": float(qm.mutual_information),
                    "inlier_ratio": float(qm.inlier_ratio),
                    "q_score": float(qm.q_score),
                    "rmse": float(qm.rmse),
                    "spatial_distribution_score": float(qm.spatial_distribution_score)
                }
                
            rmse = None
            inlier_count = None
            inlier_ratio = None
            if res.transformation is not None:
                tr = res.transformation
                rmse = float(tr.rmse)
                inlier_count = len(res.refined_matches)
                inlier_ratio = float(tr.rmse) # wait, inlier_ratio is in quality_metrics
                if q_metrics is not None:
                    inlier_ratio = q_metrics["inlier_ratio"]
                    
            result_data = {
                "success": True,
                "quality_metrics": q_metrics,
                "rmse": rmse,
                "inlier_count": inlier_count,
                "inlier_ratio": inlier_ratio,
                "execution_time_seconds": float(res.execution_time_seconds),
                "error_message": None,
                "match_points": match_points
            }
            
            job_data.update({
                "status": "completed",
                "result": result_data,
                "registered_image_url": f"/api/v1/jobs/{job_id}/image"
            })
        else:
            job_data.update({
                "status": "failed",
                "result": {
                    "success": False,
                    "execution_time_seconds": float(res.execution_time_seconds),
                    "error_message": res.error_message or "Unknown pipeline failure",
                    "match_points": []
                }
            })
    except Exception as e:
        logger.exception(f"Unhandled exception in background job {job_id}")
        job_data.update({
            "status": "failed",
            "result": {
                "success": False,
                "execution_time_seconds": 0.0,
                "error_message": f"Orchestrator failure: {e}",
                "match_points": []
            }
        })
        
    # Write final job result
    with open(job_file, "w") as f:
        json.dump(job_data, f)


@router.post("/register", response_model=JobResponse)
async def register(
    request: Request,
    background_tasks: BackgroundTasks,
):
    # Parse multipart form with a 4 GB per-part size limit to handle large .img files
    try:
        form = await request.form(max_files=10, max_fields=20, max_part_size=4 * 1024 * 1024 * 1024)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Form parse error: {e}")

    source_image: UploadFile = form.get("source_image")
    reference_image: UploadFile = form.get("reference_image")
    if source_image is None or reference_image is None:
        raise HTTPException(status_code=422, detail="source_image and reference_image are required")

    illumination_method: str = form.get("illumination_method", "phase_congruency")
    detection_method: str    = form.get("detection_method",    "superpoint")
    matching_method: str     = form.get("matching_method",     "lightglue")
    outlier_method: str      = form.get("outlier_method",      "magsac++")
    transform_type: Optional[str] = form.get("transform_type", None) or None
    refine_subpixel: bool    = str(form.get("refine_subpixel", "true")).lower() == "true"
    device: str              = form.get("device",              "auto")

    # Validate configuration parameters
    try:
        config = RegistrationConfig(
            illumination_method=illumination_method,
            detection_method=detection_method,
            matching_method=matching_method,
            outlier_method=outlier_method,
            transform_type=transform_type,
            refine_subpixel=refine_subpixel,
            device=device
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    job_id = str(uuid.uuid4().hex)
    
    # Define file paths
    src_ext = Path(source_image.filename).suffix or ".tif"
    ref_ext = Path(reference_image.filename).suffix or ".tif"
    
    src_path = UPLOADS_DIR / f"{job_id}_source{src_ext}"
    ref_path = UPLOADS_DIR / f"{job_id}_reference{ref_ext}"
    
    # Save uploaded files using chunked streaming (handles large .img files gracefully)
    CHUNK = 8 * 1024 * 1024  # 8 MB chunks
    with open(src_path, "wb") as f:
        while chunk := await source_image.read(CHUNK):
            f.write(chunk)
    with open(ref_path, "wb") as f:
        while chunk := await reference_image.read(CHUNK):
            f.write(chunk)
        
    # Initialize job data to pending
    job_data = {
        "job_id": job_id,
        "status": "pending",
        "result": None,
        "registered_image_url": None,
        "source_path": str(src_path),
        "reference_path": str(ref_path)
    }
    with open(JOBS_DIR / f"{job_id}.json", "w") as f:
        json.dump(job_data, f)
        
    # Schedule execution
    background_tasks.add_task(
        run_pipeline_job,
        job_id,
        src_path,
        ref_path,
        RESULTS_DIR,
        config
    )
    
    return JobResponse(job_id=job_id, status="pending")


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        raise HTTPException(status_code=404, detail="Job not found")
        
    with open(job_file, "r") as f:
        job_data = json.load(f)
        
    return JobResponse(
        job_id=job_data["job_id"],
        status=job_data["status"],
        result=job_data.get("result"),
        registered_image_url=job_data.get("registered_image_url")
    )


@router.get("/jobs/{job_id}/result", response_model=JobResponse)
async def get_job_result(job_id: str):
    return await get_job_status(job_id)


@router.get("/jobs/{job_id}/image")
async def download_registered_image(job_id: str):
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        raise HTTPException(status_code=404, detail="Job not found")
        
    with open(job_file, "r") as f:
        job_data = json.load(f)
        
    if job_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed yet")
        
    # Output path in results dir
    source_path = Path(job_data["source_path"])
    out_filename = f"{source_path.name}_registered.tif"
    output_path = RESULTS_DIR / out_filename
    
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Registered image file not found")
        
    return FileResponse(
        path=output_path,
        media_type="image/tiff",
        filename=out_filename
    )


@router.get("/jobs/{job_id}/matches")
async def get_job_matches_geojson(job_id: str):
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        raise HTTPException(status_code=404, detail="Job not found")
        
    with open(job_file, "r") as f:
        job_data = json.load(f)
        
    if job_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed yet")
        
    result = job_data["result"]
    if not result or not result["match_points"]:
        return {"type": "FeatureCollection", "features": []}
        
    ref_path = job_data["reference_path"]
    
    # Read reference dataset to project pixel coordinates to geospatials
    try:
        with rasterio.open(ref_path) as ref_ds:
            features = []
            for m in result["match_points"]:
                rx, ry = m["reference_pt"]
                # Convert pixel column, row to geospatial coordinate
                gx, gy = ref_ds.xy(ry, rx)
                
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(gx), float(gy)]
                    },
                    "properties": {
                        "source_pixel": m["source_pt"],
                        "reference_pixel": m["reference_pt"],
                        "confidence": float(m["confidence"])
                    }
                })
                
            return {
                "type": "FeatureCollection",
                "features": features
            }
    except Exception as e:
        logger.warning(f"Failed to georeference matches for job {job_id}: {e}")
        # Return fallback pixel-based GeoJSON (assuming coordinates represent pixels)
        features = []
        for m in result["match_points"]:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(m["reference_pt"][0]), float(m["reference_pt"][1])]
                },
                "properties": {
                    "source_pixel": m["source_pt"],
                    "reference_pixel": m["reference_pt"],
                    "confidence": float(m["confidence"])
                }
            })
        return {
            "type": "FeatureCollection",
            "features": features
        }


@router.get("/jobs/{job_id}/matches/csv")
async def download_matches_csv(job_id: str):
    """Generates downloadable CSV containing all inlier match coordinates and confidence."""
    import io
    from fastapi.responses import StreamingResponse
    
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        raise HTTPException(status_code=404, detail="Job not found")
        
    with open(job_file, "r") as f:
        job_data = json.load(f)
        
    if job_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed yet")
        
    result = job_data["result"]
    if not result or not result["match_points"]:
        raise HTTPException(status_code=400, detail="No matches found for this job")
        
    csv_io = io.StringIO()
    csv_io.write("match_id,source_x,source_y,reference_x,reference_y,confidence\n")
    for idx, m in enumerate(result["match_points"]):
        sx, sy = m["source_pt"]
        rx, ry = m["reference_pt"]
        conf = m["confidence"]
        csv_io.write(f"{idx},{sx:.3f},{sy:.3f},{rx:.3f},{ry:.3f},{conf:.4f}\n")
        
    csv_io.seek(0)
    return StreamingResponse(
        io.BytesIO(csv_io.getvalue().encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=job_{job_id}_matches.csv"}
    )


@router.get("/jobs/{job_id}/preview/{image_type}")
async def get_image_preview(job_id: str, image_type: str):
    """Generates PNG preview of source, reference, or registered GeoTIFF files."""
    import cv2
    import io
    from fastapi.responses import StreamingResponse
    
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        raise HTTPException(status_code=404, detail="Job not found")
        
    with open(job_file, "r") as f:
        job_data = json.load(f)
        
    if image_type == "source":
        img_path = Path(job_data["source_path"])
    elif image_type == "reference":
        img_path = Path(job_data["reference_path"])
    elif image_type == "registered":
        if job_data["status"] != "completed":
            raise HTTPException(status_code=400, detail="Job is not completed")
        source_path = Path(job_data["source_path"])
        img_path = RESULTS_DIR / f"{source_path.name}_registered.tif"
    else:
        raise HTTPException(status_code=400, detail="Invalid image type")
        
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")
        
    try:
        with rasterio.open(img_path) as src:
            # Read first band
            band = src.read(1)
            # Set nodata to 0
            if src.nodata is not None:
                band = np.where(band == src.nodata, 0, band)
                
            b_min, b_max = float(band.min()), float(band.max())
            if b_max > b_min:
                normalized = ((band - b_min) / (b_max - b_min) * 255.0).astype(np.uint8)
            else:
                normalized = np.zeros_like(band, dtype=np.uint8)
                
            # Resize if the image is too large for web visualization (e.g. max width/height 1024)
            h, w = normalized.shape
            max_size = 1024
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                normalized = cv2.resize(normalized, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                
            success, encoded_img = cv2.imencode('.png', normalized)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to encode PNG preview")
                
            return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
    except Exception as e:
        logger.exception(f"Failed to generate preview for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {e}")


@router.get("/health")
async def health_check():
    return {"status": "healthy"}


@router.get("/config/methods")
async def get_config_methods():
    return {
        "illumination_methods": ["phase_congruency", "clahe", "gradient", "lnms"],
        "detection_methods": ["sift", "superpoint"],
        "matching_methods": ["bf", "lightglue", "loftr"],
        "outlier_methods": ["ransac", "magsac++", "lmeds"]
    }
