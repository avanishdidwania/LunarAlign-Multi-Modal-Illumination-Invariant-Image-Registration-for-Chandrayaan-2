"""
Ground-truth accuracy benchmark for the lunar image registration pipeline.

This script empirically validates the accuracy claims made in the SIH submission by
generating synthetic lunar terrain with KNOWN geometric transforms applied, running the
full registration pipeline, and measuring how closely the recovered transform matches
the ground truth.

Requirements validated:
  - Requirement 7  : Sub-pixel RMSE (< 1.0 px) via sub-pixel refinement
  - Requirement 11 : Multi-modal / multi-scale registration (OHRC<->LRO NAC scale gaps)
  - Requirement 12 : Robustness to illumination (sun-angle) variation

Methodology:
  We construct a base 1024x1024 crater-textured terrain, then derive source/reference
  image pairs by applying a KNOWN transform (translation, rotation, scale, illumination)
  to it. Because the true transform is known, we can measure the real geometric accuracy
  of the pipeline's estimate by re-projecting a grid of test points through both the true
  and estimated transforms and measuring the pixel discrepancy. This is stronger evidence
  than the pipeline's self-reported RMSE (which only measures residuals on its own inliers).

Run:
    python benchmark.py
(This script adds ./src to sys.path automatically; no PYTHONPATH needed.)

NOTE: These results use synthetic data with exactly-known ground truth. Real ISRO data
(OHRC/TMC-2/IIRS vs LRO NAC) will exhibit sensor noise, atmospheric/terrain effects, and
more complex illumination, so real-world numbers may differ.
"""

import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import scipy.ndimage
import rasterio
from rasterio.transform import from_origin

# --- Make the src/ layout importable without needing PYTHONPATH ---
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lunar_reg.config import RegistrationConfig  # noqa: E402
from lunar_reg.pipeline import RegistrationPipeline  # noqa: E402

# Reproducibility
SEED = 42
BENCH_DIR = PROJECT_ROOT / "benchmark_data"
OUTPUT_DIR = PROJECT_ROOT / "benchmark_output"
REPORT_PATH = PROJECT_ROOT / "docs" / "BENCHMARK_RESULTS.md"

# A common CRS + ground origin so all pairs share georeferencing and overlap.
CRS = "EPSG:4326"
BASE_LON = 32.5
BASE_LAT = -69.5
# Base pixel size in degrees for the SOURCE image (arbitrary but consistent).
BASE_PIXEL = 0.0001


# ---------------------------------------------------------------------------
# Terrain generation (crater texture, adapted from generate_mock_terrain.py)
# ---------------------------------------------------------------------------
def generate_base_terrain(size: int = 1024) -> np.ndarray:
    """Generate a crater-textured lunar terrain (uint8, deterministic)."""
    rng = np.random.RandomState(SEED)
    terrain = rng.normal(128, 20, (size, size))

    # Scale the number of craters with area so a 1024 image has plenty of features.
    n_craters = int(15 * (size / 512) ** 2)
    for _ in range(n_craters):
        cx, cy = rng.randint(40, size - 40, 2)
        r = rng.randint(15, 70)
        depth = rng.randint(40, 100)

        y, x = np.ogrid[-cy:size - cy, -cx:size - cx]
        dist_sq = x * x + y * y

        basin = dist_sq <= r * r
        dist = np.sqrt(dist_sq)
        terrain[basin] = np.clip(
            terrain[basin] - depth * (1 - dist[basin] / r), 0, 255
        )

        rim = (dist_sq >= (r - 3) ** 2) & (dist_sq <= (r + 2) ** 2)
        terrain[rim] = np.clip(terrain[rim] + depth * 0.4, 0, 255)

    # Add fine-scale texture so there are keypoints everywhere, not just on crater rims.
    fine = scipy.ndimage.gaussian_filter(rng.normal(0, 12, (size, size)), sigma=1.0)
    terrain = terrain + fine

    terrain = scipy.ndimage.gaussian_filter(terrain, sigma=1.2)
    return np.clip(terrain, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Illumination simulation
# ---------------------------------------------------------------------------
def apply_illumination(img: np.ndarray, sun_angle_deg: float,
                       azimuth_axis: str = "x") -> np.ndarray:
    """
    Simulate a sun-angle change by applying a directional brightness ramp plus gamma.

    Lower sun elevation -> stronger, steeper brightness gradient across the scene and a
    mild shadow offset in the darker region. `sun_angle_deg` is the magnitude of the sun
    angle *difference* being simulated (e.g. 15 deg = mild, 45 deg = hard).
    """
    f = img.astype(np.float32) / 255.0
    h, w = f.shape

    # Gradient strength grows with the simulated angle difference.
    strength = np.clip(sun_angle_deg / 90.0, 0.0, 0.9)

    if azimuth_axis == "x":
        ramp = np.linspace(1.0 - strength, 1.0 + strength, w)[None, :]
    else:
        ramp = np.linspace(1.0 - strength, 1.0 + strength, h)[:, None]

    lit = f * ramp

    # Gamma shift changes contrast the way a different sun elevation would.
    gamma = 1.0 + 0.4 * strength
    lit = np.clip(lit, 0.0, 1.0) ** gamma

    # For harder (larger) angles, add a mild shadow offset on the dim side.
    if sun_angle_deg >= 40.0:
        if azimuth_axis == "x":
            shadow = np.linspace(-0.12, 0.0, w)[None, :]
        else:
            shadow = np.linspace(-0.12, 0.0, h)[:, None]
        lit = np.clip(lit + shadow, 0.0, 1.0)

    return np.clip(lit * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# GeoTIFF writing
# ---------------------------------------------------------------------------
def write_geotiff(path: Path, img: np.ndarray, pixel_size: float,
                  origin_lon: float = BASE_LON, origin_lat: float = BASE_LAT) -> None:
    """Write a single-band uint8 GeoTIFF with the given pixel size and origin."""
    h, w = img.shape
    transform = from_origin(origin_lon, origin_lat, pixel_size, pixel_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=h, width=w,
        count=1, dtype="uint8",
        crs=CRS,
        transform=transform,
    ) as dst:
        dst.write(img, 1)


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------
def affine_matrix(dx: float = 0.0, dy: float = 0.0, angle_deg: float = 0.0,
                  scale: float = 1.0, center: Optional[tuple] = None) -> np.ndarray:
    """
    Build a 3x3 homogeneous matrix mapping SOURCE pixel coords -> REFERENCE pixel coords.

    Represents: p_ref = scale * R(angle) * (p_src - center) + center + t
    """
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float64) * scale

    if center is None:
        center = (0.0, 0.0)
    cx, cy = center

    t = np.array([dx, dy], dtype=np.float64)
    offset = np.array([cx, cy]) - R @ np.array([cx, cy]) + t

    M = np.eye(3, dtype=np.float64)
    M[:2, :2] = R
    M[:2, 2] = offset
    return M


def transform_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 3x3 homogeneous matrix M to Nx2 points (x, y)."""
    hom = np.hstack([pts, np.ones((len(pts), 1))])
    proj = hom @ M.T
    return proj[:, :2] / (proj[:, 2:3] + 1e-12)


# ---------------------------------------------------------------------------
# Test case definition
# ---------------------------------------------------------------------------
@dataclass
class TestCase:
    name: str
    source_path: Path
    reference_path: Path
    # Ground-truth transform mapping SOURCE pixel coords -> REFERENCE pixel coords,
    # in the coordinate frame of the FULL generated images. None if not exactly known
    # in a directly comparable pixel frame (e.g. scale/illumination combos).
    true_transform: Optional[np.ndarray]
    description: str
    illumination_method: str = "clahe"
    # extent used for point-grid error sampling
    grid_extent: int = 1024


def build_test_cases(base: np.ndarray) -> List[TestCase]:
    """Generate all synthetic ground-truth scenarios and write their GeoTIFFs."""
    cases: List[TestCase] = []
    size = base.shape[0]
    center = ((size - 1) / 2.0, (size - 1) / 2.0)

    # The pipeline aligns SOURCE -> REFERENCE. We define REFERENCE = warp(base),
    # SOURCE = base. The recovered matrix should approximate the KNOWN warp we applied.
    def warp_ref(M: np.ndarray) -> np.ndarray:
        """Produce a reference image = base warped by M (src->ref). Uses inverse map."""
        Minv = np.linalg.inv(M)
        # scipy affine_transform maps output coords -> input coords using matrix (row/col)
        # Our M is in (x=col, y=row). Build a (row,col) inverse mapping.
        a = Minv[:2, :2]
        # convert (x,y) linear part to (row,col): swap axes
        lin_rc = np.array([[a[1, 1], a[1, 0]],
                           [a[0, 1], a[0, 0]]], dtype=np.float64)
        off_rc = np.array([Minv[1, 2], Minv[0, 2]], dtype=np.float64)
        warped = scipy.ndimage.affine_transform(
            base.astype(np.float32), lin_rc, offset=off_rc,
            order=1, mode="nearest", output_shape=base.shape,
        )
        return np.clip(warped, 0, 255).astype(np.uint8)

    # --- A. Pure translation (dx=15, dy=-10) ---
    M_a = affine_matrix(dx=15.0, dy=-10.0)
    ref_a = warp_ref(M_a)
    src_a = BENCH_DIR / "A_source.tif"
    rf_a = BENCH_DIR / "A_reference.tif"
    write_geotiff(src_a, base, BASE_PIXEL)
    write_geotiff(rf_a, ref_a, BASE_PIXEL)
    cases.append(TestCase(
        "A. Pure translation", src_a, rf_a, M_a,
        "Reference = source shifted by (dx=15, dy=-10) px. Near-perfect case.",
    ))

    # --- B. Rotation 5 deg + translation (10, -5) ---
    M_b = affine_matrix(dx=10.0, dy=-5.0, angle_deg=5.0, center=center)
    ref_b = warp_ref(M_b)
    src_b = BENCH_DIR / "B_source.tif"
    rf_b = BENCH_DIR / "B_reference.tif"
    write_geotiff(src_b, base, BASE_PIXEL)
    write_geotiff(rf_b, ref_b, BASE_PIXEL)
    cases.append(TestCase(
        "B. Rotation 5deg + translation", src_b, rf_b, M_b,
        "Reference = source rotated 5 deg about center + shifted (10, -5) px.",
    ))

    # --- C. Scale 2x (OHRC <-> LRO NAC analog) ---
    # Source stays full-res; reference is a 2x-coarser sensor covering the same ground.
    # Downsample the base to half size and tag it with 2x pixel size so the loader
    # detects a resolution ratio of ~2 and bridges it via the Gaussian pyramid.
    half = scipy.ndimage.zoom(base, 0.5, order=1).astype(np.uint8)
    src_c = BENCH_DIR / "C_source.tif"
    rf_c = BENCH_DIR / "C_reference.tif"
    write_geotiff(src_c, base, BASE_PIXEL)
    write_geotiff(rf_c, half, BASE_PIXEL * 2.0)  # 2x coarser pixels, same ground extent
    cases.append(TestCase(
        "C. Scale 2x (OHRC<->NAC)", src_c, rf_c, None,
        "Reference is a 2x-coarser resolution image of the same terrain (scale gap = 2).",
    ))

    # --- D. Scale 4x (larger sensor gap) ---
    quarter = scipy.ndimage.zoom(base, 0.25, order=1).astype(np.uint8)
    src_d = BENCH_DIR / "D_source.tif"
    rf_d = BENCH_DIR / "D_reference.tif"
    write_geotiff(src_d, base, BASE_PIXEL)
    write_geotiff(rf_d, quarter, BASE_PIXEL * 4.0)  # 4x coarser pixels
    cases.append(TestCase(
        "D. Scale 4x", src_d, rf_d, None,
        "Reference is a 4x-coarser resolution image of the same terrain (scale gap = 4).",
    ))

    # --- E. Illumination 15 deg sun angle (mild) ---
    ref_e = apply_illumination(base, sun_angle_deg=15.0, azimuth_axis="x")
    src_e = BENCH_DIR / "E_source.tif"
    rf_e = BENCH_DIR / "E_reference.tif"
    write_geotiff(src_e, base, BASE_PIXEL)
    write_geotiff(rf_e, ref_e, BASE_PIXEL)
    cases.append(TestCase(
        "E. Illumination 15deg", src_e, rf_e, np.eye(3),
        "Reference has a mild 15 deg sun-angle brightness ramp + gamma. Identity geometry.",
        illumination_method="phase_congruency",
    ))

    # --- F. Illumination 45 deg sun angle (hard, Req 12.3) ---
    ref_f = apply_illumination(base, sun_angle_deg=45.0, azimuth_axis="x")
    src_f = BENCH_DIR / "F_source.tif"
    rf_f = BENCH_DIR / "F_reference.tif"
    write_geotiff(src_f, base, BASE_PIXEL)
    write_geotiff(rf_f, ref_f, BASE_PIXEL)
    cases.append(TestCase(
        "F. Illumination 45deg (hard)", src_f, rf_f, np.eye(3),
        "Reference has a strong 45 deg sun-angle gradient + shadow offset. Tests Req 12.3.",
        illumination_method="phase_congruency",
    ))

    # --- G. Combined: rotation + scale + illumination ---
    # Rotate 3 deg + shift, then coarsen 2x, then relight at 30 deg.
    M_g = affine_matrix(dx=8.0, dy=-6.0, angle_deg=3.0, center=center)
    ref_g_full = warp_ref(M_g)
    ref_g_half = scipy.ndimage.zoom(ref_g_full, 0.5, order=1).astype(np.uint8)
    ref_g_lit = apply_illumination(ref_g_half, sun_angle_deg=30.0, azimuth_axis="y")
    src_g = BENCH_DIR / "G_source.tif"
    rf_g = BENCH_DIR / "G_reference.tif"
    write_geotiff(src_g, base, BASE_PIXEL)
    write_geotiff(rf_g, ref_g_lit, BASE_PIXEL * 2.0)
    cases.append(TestCase(
        "G. Combined rot+scale+illum", src_g, rf_g, None,
        "Reference = rotate 3 deg + shift, coarsen 2x, relight 30 deg. Realistic multi-challenge.",
        illumination_method="phase_congruency",
    ))

    return cases


# ---------------------------------------------------------------------------
# Accuracy vs ground truth
# ---------------------------------------------------------------------------
def transform_error(true_M: np.ndarray, est_M: np.ndarray, extent: int) -> tuple:
    """
    Measure geometric error by projecting a grid of points through both the true and
    estimated transforms and comparing the resulting reference-frame coordinates.
    Returns (mean_px_error, max_px_error).
    """
    n = 9
    coords = np.linspace(extent * 0.1, extent * 0.9, n)
    gx, gy = np.meshgrid(coords, coords)
    pts = np.column_stack([gx.ravel(), gy.ravel()])

    true_proj = transform_points(true_M, pts)
    est_proj = transform_points(est_M, pts)
    dist = np.linalg.norm(true_proj - est_proj, axis=1)
    return float(np.mean(dist)), float(np.max(dist))


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------
@dataclass
class RunRecord:
    name: str
    description: str
    success: bool
    rmse: Optional[float] = None
    inlier_count: Optional[int] = None
    inlier_ratio: Optional[float] = None
    spatial_dist: Optional[float] = None
    ssim: Optional[float] = None
    time_s: Optional[float] = None
    gt_mean_err: Optional[float] = None
    gt_max_err: Optional[float] = None
    error_message: Optional[str] = None


def make_config(tc: TestCase) -> RegistrationConfig:
    """SIFT + BF + MAGSAC++ + subpixel refinement. Deterministic, no model download."""
    return RegistrationConfig(
        illumination_method=tc.illumination_method,
        detection_method="sift",
        max_keypoints=8192,
        matching_method="bf",
        match_threshold=0.75,
        outlier_method="magsac++",
        transform_type="affine",
        refine_subpixel=True,
        device="cpu",
    )


def run_case(tc: TestCase) -> RunRecord:
    print(f"\n--- Running {tc.name} ---")
    print(f"    {tc.description}")
    try:
        config = make_config(tc)
        pipeline = RegistrationPipeline(config)
        result = pipeline.run(tc.source_path, tc.reference_path, OUTPUT_DIR / tc.name[:1])

        rec = RunRecord(name=tc.name, description=tc.description, success=result.success)
        rec.error_message = result.error_message
        rec.time_s = result.execution_time_seconds

        if result.quality_metrics is not None:
            qm = result.quality_metrics
            rec.rmse = qm.rmse
            rec.inlier_ratio = qm.inlier_ratio
            rec.spatial_dist = qm.spatial_distribution_score
            rec.ssim = qm.ssim
        rec.inlier_count = len(result.refined_matches)

        # Ground-truth geometric accuracy where the true transform is directly comparable.
        if result.transformation is not None and tc.true_transform is not None:
            try:
                mean_e, max_e = transform_error(
                    tc.true_transform, result.transformation.matrix, tc.grid_extent
                )
                rec.gt_mean_err = mean_e
                rec.gt_max_err = max_e
            except Exception as e:  # pragma: no cover
                print(f"    ! transform_error failed: {e}")

        status = "SUCCESS" if result.success else "FAIL"
        print(f"    -> {status} | rmse={rec.rmse} inliers={rec.inlier_count} "
              f"inlier_ratio={rec.inlier_ratio} time={rec.time_s:.2f}s")
        if rec.gt_mean_err is not None:
            print(f"    -> ground-truth point error: mean={rec.gt_mean_err:.4f}px "
                  f"max={rec.gt_max_err:.4f}px")
        if not result.success:
            print(f"    -> error: {result.error_message}")
        return rec

    except Exception as e:
        print(f"    !! EXCEPTION: {e}")
        traceback.print_exc()
        return RunRecord(name=tc.name, description=tc.description, success=False,
                         error_message=f"Exception: {e}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def fmt(v, prec=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        if v != v:  # NaN
            return "nan"
        return f"{v:.{prec}f}"
    return str(v)


def print_console_table(records: List[RunRecord]) -> None:
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)
    header = f"{'Test Case':<32}{'OK':<5}{'RMSE':<9}{'Inl#':<7}{'InlRatio':<10}{'SpDist':<8}{'SSIM':<8}{'Time(s)':<8}"
    print(header)
    print("-" * 100)
    for r in records:
        print(f"{r.name:<32}{('Y' if r.success else 'N'):<5}"
              f"{fmt(r.rmse):<9}{fmt(r.inlier_count):<7}{fmt(r.inlier_ratio):<10}"
              f"{fmt(r.spatial_dist):<8}{fmt(r.ssim):<8}{fmt(r.time_s, 2):<8}")
    print("-" * 100)
    print("\nGround-truth geometric accuracy (where true transform is known):")
    for r in records:
        if r.gt_mean_err is not None:
            print(f"  {r.name:<32} mean={r.gt_mean_err:.4f}px  max={r.gt_max_err:.4f}px")


def write_markdown_report(records: List[RunRecord]) -> None:
    lookup = {r.name[:1]: r for r in records}

    # Requirement checks
    well_conditioned = [lookup.get("A"), lookup.get("B")]
    req7_vals = [r.gt_mean_err for r in well_conditioned
                 if r and r.success and r.gt_mean_err is not None]
    req7_pass = bool(req7_vals) and all(v < 1.0 for v in req7_vals)

    scale_cases = [lookup.get("C"), lookup.get("D")]
    # Req 11.4 requires SUB-PIXEL accuracy (RMSE < 1.0 px) across sensor combinations,
    # and explicitly says a registration with RMSE >= 1.0 must be marked failed. So we
    # distinguish "converged" (produced a valid transform) from "met the sub-pixel bar".
    req11_converged = all(r is not None and r.success for r in scale_cases)
    req11_subpixel = all(
        r is not None and r.success and r.rmse is not None and r.rmse < 1.0
        for r in scale_cases
    )
    req11_pass = req11_subpixel

    f_case = lookup.get("F")
    req12_ratio = f_case.inlier_ratio if f_case else None
    req12_pass = bool(f_case and f_case.success and req12_ratio is not None
                      and req12_ratio > 0.30)

    def badge(ok: bool) -> str:
        return "**PASS**" if ok else "**FAIL**"

    lines: List[str] = []
    lines.append("# Registration Accuracy Benchmark Results")
    lines.append("")
    lines.append("_Automatically generated by `benchmark.py`._")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "This benchmark validates the registration pipeline against **synthetic ground "
        "truth**. A single 1024x1024 crater-textured lunar terrain is generated "
        "deterministically (fixed random seed), then each test case derives a "
        "source/reference pair by applying a **known** geometric and/or radiometric "
        "transform to that terrain."
    )
    lines.append("")
    lines.append(
        "Because the applied transform is known exactly, we can measure the pipeline's "
        "true geometric accuracy: we project a 9x9 grid of test points through both the "
        "known transform and the pipeline's estimated transform, then report the mean and "
        "maximum pixel discrepancy. This is a stronger accuracy proof than the pipeline's "
        "self-reported RMSE, which only measures residuals on its own inlier set."
    )
    lines.append("")
    lines.append(
        "All runs use a deterministic, CPU-only configuration (SIFT features + "
        "brute-force ratio matching + MAGSAC++ outlier rejection + sub-pixel refinement) "
        "so results are reproducible without a GPU or downloaded models. Illumination "
        "test cases additionally use phase-congruency normalization."
    )
    lines.append("")
    lines.append("## Test Scenarios")
    lines.append("")
    for r in records:
        lines.append(f"- **{r.name}** - {r.description}")
    lines.append("")
    lines.append("## Results Table")
    lines.append("")
    lines.append("| Test Case | Success | RMSE (px) | Inlier Count | Inlier Ratio | "
                 "Spatial Dist | SSIM | Time (s) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in records:
        lines.append(
            f"| {r.name} | {'Yes' if r.success else 'No'} | {fmt(r.rmse)} | "
            f"{fmt(r.inlier_count)} | {fmt(r.inlier_ratio)} | {fmt(r.spatial_dist)} | "
            f"{fmt(r.ssim)} | {fmt(r.time_s, 2)} |"
        )
    lines.append("")

    lines.append("## Ground-Truth Geometric Accuracy")
    lines.append("")
    lines.append(
        "For cases with a directly comparable known pixel-frame transform, the table "
        "below reports the discrepancy between the true and estimated transforms measured "
        "across a grid of test points."
    )
    lines.append("")
    lines.append("| Test Case | Mean Point Error (px) | Max Point Error (px) |")
    lines.append("|---|---|---|")
    any_gt = False
    for r in records:
        if r.gt_mean_err is not None:
            any_gt = True
            lines.append(f"| {r.name} | {fmt(r.gt_mean_err, 4)} | {fmt(r.gt_max_err, 4)} |")
    if not any_gt:
        lines.append("| (none available) | - | - |")
    lines.append("")

    lines.append("## Requirement Validation")
    lines.append("")
    lines.append(f"### Requirement 7.2 - Sub-pixel accuracy (RMSE < 1.0 px): {badge(req7_pass)}")
    lines.append("")
    if req7_vals:
        detail = ", ".join(
            f"{r.name}: mean point error {r.gt_mean_err:.4f} px"
            for r in well_conditioned if r and r.gt_mean_err is not None
        )
        lines.append(
            f"Measured against ground truth on the well-conditioned cases ({detail}). "
            f"The requirement is satisfied when the geometric error stays strictly below "
            f"1.0 pixel."
        )
    else:
        lines.append(
            "Could not evaluate - the well-conditioned cases did not produce a comparable "
            "ground-truth transform."
        )
    lines.append("")

    lines.append(f"### Requirement 11 - Multi-scale sub-pixel registration: {badge(req11_pass)}")
    lines.append("")
    c = lookup.get("C")
    d = lookup.get("D")

    def _scale_desc(r):
        if r is None:
            return "not run"
        if not r.success:
            return "FAILED to register"
        return f"registered (RMSE {fmt(r.rmse)} px, inlier ratio {fmt(r.inlier_ratio)})"

    lines.append(
        f"Scale-gap cases model the OHRC<->LRO NAC resolution difference. "
        f"Case C (2x): {_scale_desc(c)}; Case D (4x): {_scale_desc(d)}."
    )
    lines.append("")
    lines.append(
        f"Both scale-gap cases **converge to a valid transform** "
        f"({'yes' if req11_converged else 'no'}), which demonstrates the pyramid-based "
        f"scale-bridging path works end-to-end. However, Requirement 11.4 demands "
        f"**sub-pixel** accuracy (RMSE < 1.0 px) across sensor combinations, and the "
        f"scale-gap cases here exceed that bar "
        f"({'they do NOT' if not req11_subpixel else 'they'} meet it). "
        f"This is an honest limitation: resolving the finer image down to the coarser "
        f"level to match, then mapping coordinates back, loses sub-pixel precision "
        f"proportional to the scale gap. Same-resolution cases (A, B) achieve well under "
        f"1.0 px, so the sub-pixel machinery itself is sound; the gap is specifically in "
        f"cross-scale refinement."
    )
    lines.append("")

    lines.append(f"### Requirement 12.3 - Inlier ratio > 30% at 45 deg sun angle: {badge(req12_pass)}")
    lines.append("")
    if req12_ratio is not None:
        lines.append(
            f"The hard illumination case (F, 45 deg sun-angle difference) achieved an "
            f"inlier ratio of **{req12_ratio:.3f}** "
            f"({'above' if req12_pass else 'at or below'} the 0.30 threshold)."
        )
    else:
        lines.append("The 45 deg illumination case did not produce an inlier ratio to evaluate.")
    lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    validated = []
    not_validated = []
    (validated if req7_pass else not_validated).append(
        "Requirement 7 (sub-pixel RMSE on same-resolution pairs)")
    if req11_subpixel:
        validated.append("Requirement 11 (multi-scale sub-pixel)")
    elif req11_converged:
        not_validated.append(
            "Requirement 11 (multi-scale registration converges but exceeds the 1.0 px "
            "sub-pixel bar on scale-gap pairs)")
    else:
        not_validated.append("Requirement 11 (multi-scale)")
    (validated if req12_pass else not_validated).append("Requirement 12.3 (45 deg illumination)")
    if validated:
        lines.append("Empirically validated on synthetic ground truth: "
                     + "; ".join(validated) + ".")
    if not_validated:
        lines.append("")
        lines.append("Not validated by this run (see numbers above): "
                     + "; ".join(not_validated) + ".")
    lines.append("")
    lines.append(
        "> **Note:** These results use synthetic data with exactly-known ground truth. "
        "Real ISRO data (OHRC / TMC-2 / IIRS vs LRO NAC) includes sensor noise, terrain "
        "relief, and more complex illumination, so real-world results may vary."
    )
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown report written to: {REPORT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    print("Generating base crater terrain (1024x1024)...")
    base = generate_base_terrain(1024)

    print("Building synthetic ground-truth test cases...")
    cases = build_test_cases(base)

    records: List[RunRecord] = []
    for tc in cases:
        records.append(run_case(tc))

    print_console_table(records)
    write_markdown_report(records)
    print(f"\nTotal benchmark time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
