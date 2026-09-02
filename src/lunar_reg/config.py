import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

VALID_ILLUMINATION_METHODS = {"phase_congruency", "clahe", "gradient", "lnms"}
VALID_DETECTION_METHODS = {"sift", "superpoint"}
VALID_MATCHING_METHODS = {"bf", "lightglue", "loftr"}
VALID_OUTLIER_METHODS = {"ransac", "magsac++", "lmeds"}
VALID_TRANSFORM_TYPES = {None, "affine", "projective"}
VALID_INTERPOLATION_METHODS = {"bilinear", "bicubic", "lanczos"}
VALID_DEVICES = {"auto", "cuda", "cpu"}

@dataclass
class RegistrationConfig:
    illumination_method: str = "phase_congruency"
    pyramid_levels: Optional[int] = None
    pyramid_scale_factor: float = 0.5
    detection_method: str = "superpoint"
    max_keypoints: int = 8192
    matching_method: str = "lightglue"
    match_threshold: float = 0.2
    outlier_method: str = "magsac++"
    ransac_confidence: float = 0.999
    ransac_max_iters: int = 10000
    transform_type: Optional[str] = None
    refine_subpixel: bool = True
    refinement_patch_size: int = 21
    full_res_refine: bool = True
    interpolation: str = "bicubic"
    device: str = "auto"

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.illumination_method not in VALID_ILLUMINATION_METHODS:
            raise ValueError(
                f"Invalid illumination_method '{self.illumination_method}'. "
                f"Must be one of {VALID_ILLUMINATION_METHODS}"
            )
        if self.detection_method not in VALID_DETECTION_METHODS:
            raise ValueError(
                f"Invalid detection_method '{self.detection_method}'. "
                f"Must be one of {VALID_DETECTION_METHODS}"
            )
        if self.matching_method not in VALID_MATCHING_METHODS:
            raise ValueError(
                f"Invalid matching_method '{self.matching_method}'. "
                f"Must be one of {VALID_MATCHING_METHODS}"
            )
        if self.outlier_method not in VALID_OUTLIER_METHODS:
            raise ValueError(
                f"Invalid outlier_method '{self.outlier_method}'. "
                f"Must be one of {VALID_OUTLIER_METHODS}"
            )
        if self.transform_type not in VALID_TRANSFORM_TYPES:
            raise ValueError(
                f"Invalid transform_type '{self.transform_type}'. "
                f"Must be one of {VALID_TRANSFORM_TYPES}"
            )
        if self.interpolation not in VALID_INTERPOLATION_METHODS:
            raise ValueError(
                f"Invalid interpolation '{self.interpolation}'. "
                f"Must be one of {VALID_INTERPOLATION_METHODS}"
            )
        if self.device not in VALID_DEVICES:
            raise ValueError(
                f"Invalid device '{self.device}'. "
                f"Must be one of {VALID_DEVICES}"
            )
        if self.pyramid_scale_factor <= 0 or self.pyramid_scale_factor >= 1:
            raise ValueError("pyramid_scale_factor must be between 0 and 1 (exclusive)")
        if self.max_keypoints <= 0:
            raise ValueError("max_keypoints must be positive")
        if not (0 <= self.match_threshold <= 1):
            raise ValueError("match_threshold must be between 0 and 1")
        if not (0 < self.ransac_confidence < 1):
            raise ValueError("ransac_confidence must be between 0 and 1 (exclusive)")
        if self.ransac_max_iters <= 0:
            raise ValueError("ransac_max_iters must be positive")
        if self.refinement_patch_size <= 0 or self.refinement_patch_size % 2 == 0:
            raise ValueError("refinement_patch_size must be a positive odd integer")
        if not isinstance(self.full_res_refine, bool):
            raise ValueError("full_res_refine must be a boolean")

@dataclass
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    upload_max_size_mb: int = 500
    results_dir: str = "./results"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

@dataclass
class StorageConfig:
    upload_dir: str = "./uploads"
    results_dir: str = "./results"
    export_dir: str = "./exports"

@dataclass
class AppConfig:
    pipeline: RegistrationConfig = field(default_factory=RegistrationConfig)
    web: WebConfig = field(default_factory=WebConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    @classmethod
    def load_from_yaml(cls, path: Path) -> "AppConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        
        pipeline_data = data.get("pipeline", {})
        web_data = data.get("web", {})
        storage_data = data.get("storage", {})

        pipeline = RegistrationConfig(**pipeline_data)
        web = WebConfig(**web_data)
        storage = StorageConfig(**storage_data)

        return cls(pipeline=pipeline, web=web, storage=storage)
