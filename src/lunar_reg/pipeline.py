import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import rasterio

from lunar_reg.config import RegistrationConfig
from lunar_reg.loader.image_loader import ImageLoader, ImageMetadata, SensorType
from lunar_reg.preprocessing.illumination import IlluminationNormalizer
from lunar_reg.preprocessing.pyramid import PyramidBuilder
from lunar_reg.detection.sift_detector import SIFTDetector
from lunar_reg.detection.superpoint import SuperPointDetector
from lunar_reg.matching.bf_matcher import BFMatcher
from lunar_reg.matching.lightglue_matcher import LightGlueMatcher
from lunar_reg.matching.loftr_matcher import LoFTRMatcher
from lunar_reg.outlier.ransac import OutlierRejector, RobustMethod
from lunar_reg.transform.estimator import TransformationEstimator, TransformationType, TransformationResult
from lunar_reg.refinement.subpixel import SubPixelRefiner, RefinedMatch, RefinementResult
from lunar_reg.warping.warper import ImageWarper
from lunar_reg.evaluation.assessor import QualityAssessor, QualityMetrics
from lunar_reg.exporter.geotiff import GeoTiffExporter
from lunar_reg.errors import PipelineError, StageResult
from lunar_reg.matching.base import MatchPair

logger = logging.getLogger(__name__)

@dataclass
class RegistrationResult:
    success: bool
    config: RegistrationConfig
    source_metadata: ImageMetadata
    reference_metadata: ImageMetadata
    quality_metrics: Optional[QualityMetrics]
    transformation: Optional[TransformationResult]
    refined_matches: List[RefinedMatch]
    warped_image_path: Optional[Path]
    export_paths: Dict[str, Path]
    error_message: Optional[str]
    execution_time_seconds: float

class RegistrationPipeline:
    """Orchestrates the full lunar image registration pipeline."""

    def __init__(self, config: RegistrationConfig):
        self.config = config
        self._init_components()

    def _init_components(self):
        """Initialize all components based on configurations."""
        self.loader = ImageLoader()
        self.normalizer = IlluminationNormalizer()
        self.pyramid_builder = PyramidBuilder()
        
        # Initialize detector
        if self.config.detection_method == "sift":
            self.detector = SIFTDetector(n_features=self.config.max_keypoints)
        elif self.config.detection_method == "superpoint":
            self.detector = SuperPointDetector(max_keypoints=self.config.max_keypoints, device=self.config.device)
            
        # Initialize matcher
        if self.config.matching_method == "bf":
            self.matcher = BFMatcher(ratio_threshold=self.config.match_threshold, norm_type="L2")
        elif self.config.matching_method == "lightglue":
            self.matcher = LightGlueMatcher(
                features="superpoint",
                device=self.config.device,
                match_threshold=self.config.match_threshold
            )
        elif self.config.matching_method == "loftr":
            self.matcher = LoFTRMatcher(
                pretrained="outdoor",
                device=self.config.device,
                match_threshold=self.config.match_threshold
            )
            
        # Initialize outlier rejector
        method_map = {
            "ransac": RobustMethod.RANSAC,
            "magsac++": RobustMethod.MAGSAC_PLUS,
            "lmeds": RobustMethod.LMEDS
        }
        robust_method = method_map.get(self.config.outlier_method, RobustMethod.MAGSAC_PLUS)
        self.rejector = OutlierRejector(
            method=robust_method,
            confidence=self.config.ransac_confidence,
            max_iterations=self.config.ransac_max_iters,
            threshold=3.0
        )
        
        # Initialize estimator, refiner, warper, assessor, and exporter
        self.estimator = TransformationEstimator()
        self.refiner = SubPixelRefiner(
            patch_size=self.config.refinement_patch_size,
            min_correlation=0.7
        )
        self.warper = ImageWarper()
        self.assessor = QualityAssessor()
        self.exporter = GeoTiffExporter()

    def run(self, source_path: Path, reference_path: Path,
            output_dir: Path) -> RegistrationResult:
        """
        Execute full registration pipeline.
        Deterministic: same inputs + config => same outputs.
        """
        start_time = time.time()
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        def dummy_meta(path: Path) -> ImageMetadata:
            return ImageMetadata(
                file_path=Path(path),
                sensor_type=SensorType.UNKNOWN,
                spatial_resolution=1.0,
                width=0,
                height=0,
                num_bands=1,
                crs=None,
                geotransform=None,
                sun_elevation=None,
                sun_azimuth=None,
                acquisition_time=None
            )
            
        src_meta = dummy_meta(source_path)
        ref_meta = dummy_meta(reference_path)
        
        try:
            # 1. Image Loading
            logger.info("Pipeline Step 1: Loading images...")
            try:
                src_loaded, ref_loaded = self.loader.load_overlapping_pair(source_path, reference_path)
            except Exception as e:
                return RegistrationResult(
                    success=False,
                    config=self.config,
                    source_metadata=src_meta,
                    reference_metadata=ref_meta,
                    quality_metrics=None,
                    transformation=None,
                    refined_matches=[],
                    warped_image_path=None,
                    export_paths={},
                    error_message=f"Loading stage failed: {e}",
                    execution_time_seconds=time.time() - start_time
                )
                
            src_meta = src_loaded.metadata
            ref_meta = ref_loaded.metadata
            
            src_img = src_loaded.data
            ref_img = ref_loaded.data
            
            # Extract mono band for feature matching
            src_mono = src_img[0] if src_img.ndim == 3 else src_img
            ref_mono = ref_img[0] if ref_img.ndim == 3 else ref_img
            
            # 2. Illumination Normalization
            logger.info(f"Pipeline Step 2: Running illumination normalization ({self.config.illumination_method})...")
            src_norm = self.normalizer.normalize(src_mono, method=self.config.illumination_method)
            ref_norm = self.normalizer.normalize(ref_mono, method=self.config.illumination_method)
            
            # Clip and cast to uint8 if normalized to float
            if src_norm.dtype != np.uint8:
                src_norm = np.clip(src_norm * 255.0, 0, 255).astype(np.uint8)
            if ref_norm.dtype != np.uint8:
                ref_norm = np.clip(ref_norm * 255.0, 0, 255).astype(np.uint8)
                
            # 3. Resolution pyramid scale-gap bridging
            res_src = src_meta.spatial_resolution
            res_ref = ref_meta.spatial_resolution
            
            src_matching_img = src_norm
            ref_matching_img = ref_norm
            scale_factor_src = 1.0
            scale_factor_ref = 1.0
            
            ratio = max(res_src, res_ref) / (min(res_src, res_ref) + 1e-8)
            if ratio > 1.2:
                logger.info(f"Scale gap detected (ratio: {ratio:.2f}). Constructing Gaussian pyramids...")
                src_pyr = self.pyramid_builder.build(src_norm, n_levels=5, scale_factor=self.config.pyramid_scale_factor)
                ref_pyr = self.pyramid_builder.build(ref_norm, n_levels=5, scale_factor=self.config.pyramid_scale_factor)
                
                lvl_src, lvl_ref = self.pyramid_builder.find_matching_levels(
                    src_pyr, ref_pyr, res_src, res_ref
                )
                logger.info(f"Matched levels: source level={lvl_src}, reference level={lvl_ref}")
                src_matching_img = src_pyr.levels[lvl_src]
                ref_matching_img = ref_pyr.levels[lvl_ref]
                scale_factor_src = src_pyr.scale_factors[lvl_src]
                scale_factor_ref = ref_pyr.scale_factors[lvl_ref]
                
            # 4. Feature Matching
            logger.info(f"Pipeline Step 4: Establishing correspondences using {self.config.matching_method}...")
            if self.config.matching_method == "loftr":
                matching_res = self.matcher.match_images(src_matching_img, ref_matching_img)
                raw_matches = matching_res.matches
            else:
                src_det = self.detector.detect(src_matching_img)
                ref_det = self.detector.detect(ref_matching_img)
                matching_res = self.matcher.match(src_det, ref_det)
                raw_matches = matching_res.matches
                
            # Map matches back to original unscaled coordinates
            scaled_matches = []
            for m in raw_matches:
                scaled_matches.append(
                    MatchPair(
                        source_idx=m.source_idx,
                        reference_idx=m.reference_idx,
                        source_pt=(m.source_pt[0] / scale_factor_src, m.source_pt[1] / scale_factor_src),
                        reference_pt=(m.reference_pt[0] / scale_factor_ref, m.reference_pt[1] / scale_factor_ref),
                        confidence=m.confidence
                    )
                )
                
            total_initial_matches = len(scaled_matches)
            logger.info(f"Found {total_initial_matches} initial match points.")
            
            # 5. Outlier Rejection
            logger.info(f"Pipeline Step 5: Rejecting outliers using {self.config.outlier_method}...")
            t_type = self.config.transform_type
            if t_type is None:
                t_type = "homography" if self.estimator.auto_select_model(scaled_matches) == TransformationType.PROJECTIVE else "affine"
                
            model_type = "homography" if t_type == "projective" else "affine"
            outlier_res = self.rejector.reject(scaled_matches, model_type=model_type)
            final_inliers = outlier_res.inlier_matches
            logger.info(f"Outlier rejection completed: {len(final_inliers)} inliers remaining.")
            
            # 6. Sub-Pixel Refinement
            refined_list = []
            if self.config.refine_subpixel and len(final_inliers) >= 3:
                logger.info("Pipeline Step 6: Performing sub-pixel keypoint refinement...")
                refinement_res = self.refiner.refine(src_mono, ref_mono, final_inliers)
                refined_list = refinement_res.refined_matches
                
                # Re-construct refined MatchPair objects
                refined_pairs = []
                for i, rm in enumerate(refined_list):
                    refined_pairs.append(
                        MatchPair(
                            source_idx=i,
                            reference_idx=i,
                            source_pt=rm.source_pt,
                            reference_pt=rm.reference_pt,
                            confidence=rm.ncc_score
                        )
                    )
                final_inliers = refined_pairs
                logger.info(f"Sub-pixel refinement completed: {len(final_inliers)} matches refined.")
            else:
                # Populate refined_list with unrefined inliers so matches are always returned
                refined_list = [
                    RefinedMatch(
                        source_pt=m.source_pt,
                        reference_pt=m.reference_pt,
                        accuracy_estimate=0.5,
                        ncc_score=m.confidence
                    )
                    for m in final_inliers
                ]
                
            # 7. Transformation Estimation & Validation
            logger.info("Pipeline Step 7: Estimating final transformation...")
            transform_enum = TransformationType.PROJECTIVE if model_type == "homography" else TransformationType.AFFINE
            
            try:
                transform_res = self.estimator.estimate(final_inliers, transform_enum)
            except Exception as e:
                return RegistrationResult(
                    success=False,
                    config=self.config,
                    source_metadata=src_meta,
                    reference_metadata=ref_meta,
                    quality_metrics=None,
                    transformation=None,
                    refined_matches=refined_list,
                    warped_image_path=None,
                    export_paths={},
                    error_message=f"Transformation estimation stage failed: {e}",
                    execution_time_seconds=time.time() - start_time
                )
                
            is_valid = self.estimator.validate_transform(transform_res.matrix)
            if not is_valid:
                return RegistrationResult(
                    success=False,
                    config=self.config,
                    source_metadata=src_meta,
                    reference_metadata=ref_meta,
                    quality_metrics=None,
                    transformation=transform_res,
                    refined_matches=refined_list,
                    warped_image_path=None,
                    export_paths={},
                    error_message="Estimated transformation failed physical plausibility checks.",
                    execution_time_seconds=time.time() - start_time
                )
                
            # 8. Image Warping
            logger.info("Pipeline Step 8: Warping source image bands...")
            warped_mono = self.warper.warp(
                src_mono,
                transform_res.matrix,
                ref_mono.shape,
                interpolation=self.config.interpolation
            )
            
            # 9. Quality Assessment
            logger.info("Pipeline Step 9: Performing quality assessment...")
            quality_metrics = self.assessor.assess(
                warped_mono,
                ref_mono,
                final_inliers,
                total_initial_matches,
                rmse=float(transform_res.rmse) if transform_res else 0.0
            )
            
            # 10. GeoTIFF Exporting
            logger.info("Pipeline Step 10: Exporting registered bands to GeoTIFF...")
            out_filename = f"{source_path.name}_registered.tif"
            output_path = output_dir / out_filename
            
            with rasterio.open(reference_path) as ref_ds:
                ref_profile = ref_ds.profile
                
            # Warp multi-band data if present
            if src_img.ndim == 3:
                warped_bands = []
                # Handle channel dimension position
                if src_img.shape[0] <= 4:
                    num_channels = src_img.shape[0]
                    for c in range(num_channels):
                        warped_band = self.warper.warp(
                            src_img[c],
                            transform_res.matrix,
                            ref_mono.shape,
                            interpolation=self.config.interpolation
                        )
                        warped_bands.append(warped_band)
                else:
                    num_channels = src_img.shape[2]
                    for c in range(num_channels):
                        warped_band = self.warper.warp(
                            src_img[:, :, c],
                            transform_res.matrix,
                            ref_mono.shape,
                            interpolation=self.config.interpolation
                        )
                        warped_bands.append(warped_band)
                warped_multiband = np.stack(warped_bands, axis=0)
                self.exporter.export_registered(warped_multiband, str(output_path), ref_profile)
            else:
                self.exporter.export_registered(warped_mono, str(output_path), ref_profile)
                
            export_paths = {"registered_image": output_path}
            execution_time = time.time() - start_time
            logger.info(f"Pipeline completed successfully in {execution_time:.2f} seconds.")
            
            return RegistrationResult(
                success=True,
                config=self.config,
                source_metadata=src_meta,
                reference_metadata=ref_meta,
                quality_metrics=quality_metrics,
                transformation=transform_res,
                refined_matches=refined_list,
                warped_image_path=output_path,
                export_paths=export_paths,
                error_message=None,
                execution_time_seconds=execution_time
            )
            
        except Exception as e:
            logger.exception("Pipeline run encountered unhandled exception")
            return RegistrationResult(
                success=False,
                config=self.config,
                source_metadata=src_meta,
                reference_metadata=ref_meta,
                quality_metrics=None,
                transformation=None,
                refined_matches=[],
                warped_image_path=None,
                export_paths={},
                error_message=f"Unhandled exception in pipeline: {e}",
                execution_time_seconds=time.time() - start_time
            )
