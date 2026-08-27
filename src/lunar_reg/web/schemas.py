from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class RegistrationRequest(BaseModel):
    illumination_method: str = Field(default="phase_congruency", description="Illumination normalization method")
    detection_method: str = Field(default="superpoint", description="Keypoint detector (sift or superpoint)")
    matching_method: str = Field(default="lightglue", description="Feature matcher (bf, lightglue, or loftr)")
    outlier_method: str = Field(default="magsac++", description="Outlier rejection method")
    transform_type: Optional[str] = Field(default=None, description="Transformation model (affine or projective)")
    refine_subpixel: bool = Field(default=True, description="Enable sub-pixel refinement")
    device: str = Field(default="auto", description="Processing device (auto, cuda, cpu)")

class QualityMetricsSchema(BaseModel):
    ssim: float
    psnr: float
    mutual_information: float
    inlier_ratio: float
    q_score: float

class MatchPointSchema(BaseModel):
    source_pt: List[float] = Field(..., description="[x, y] coordinate in source image")
    reference_pt: List[float] = Field(..., description="[x, y] coordinate in reference image")
    confidence: float

class RegistrationResultSchema(BaseModel):
    success: bool
    quality_metrics: Optional[QualityMetricsSchema] = None
    rmse: Optional[float] = None
    inlier_count: Optional[int] = None
    inlier_ratio: Optional[float] = None
    execution_time_seconds: float
    error_message: Optional[str] = None
    match_points: List[MatchPointSchema] = Field(default_factory=list)

class JobResponse(BaseModel):
    job_id: str
    status: str  # "pending" | "running" | "completed" | "failed"
    result: Optional[RegistrationResultSchema] = None
    registered_image_url: Optional[str] = None
