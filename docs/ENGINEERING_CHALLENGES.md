# Engineering Challenges & Solutions

This document chronicles the production-grade challenges encountered while testing LunarAlign on real Chandrayaan-2 (ISRO) and LRO NAC (NASA) data. Unlike synthetic benchmarks, real planetary data surfaced format, memory, network, and numerical issues that required deep engineering solutions.

---

## 1. Planetary Data Format & Ingestion

### Challenge: "not recognized as being in a supported file format"

**Root Cause:** ISRO Chandrayaan TMC `.img` files follow the PDS4 standard (raw binary raster with geospatial metadata in a detached `.xml` label file), while NASA LRO NAC images use PDS3 (embedded headers or detached `.lbl` files). GDAL/Rasterio cannot parse raw PDS4 `.img` files without referencing their companion `.xml` metadata.

**Resolution:**

- Created `convert_pds_to_tiff.py` to parse PDS4 XML labels and extract calibrated data into standard, self-contained GeoTIFFs.
- Added automatic `.zip` bundle extraction in the backend (`routes.py`), allowing users to drag-and-drop raw downloaded archive packages directly.

---

## 2. Network, Proxy & Large Payload Failures

### Challenge A: Starlette Multipart Upload Size Limits

**Symptom:** Immediate 400 Bad Request / "Failed to submit job" on files > 1 MB.

**Root Cause:** Default Starlette/FastAPI multipart parsers enforce strict in-memory chunk limits.

**Resolution:** Implemented custom `LargeUploadMiddleware`, patched `MultiPartParser.max_file_size` to 4 GB, and converted file writes to chunked disk streams (8 MB buffers).

### Challenge B: Connection Drops (`net::ERR_CONNECTION_RESET`)

**Root Cause:** The React frontend routed requests through the Vite dev-server proxy (`/api`). Node.js proxies frequently terminate TCP sockets when proxying multi-hundred-megabyte streams over localhost.

**Resolution:** Created `frontend/.env` and `frontend/.env.development` to route API calls directly to Uvicorn at `http://localhost:8000/api/v1`, bypassing Node.js proxy buffers.

### Challenge C: Browser CORS Wildcard Rejection

**Root Cause:** FastAPI CORS middleware had both `allow_origins=["*"]` and `allow_credentials=True`. Browser security specs block responses when credentials are enabled alongside wildcard origins.

**Resolution:** Set `allow_credentials=False` in `app.py` since the API is stateless.

---

## 3. Out-Of-Memory (OOM) & Extreme Track Dimensions

### Challenge A: Phase Congruency Array Allocation Crash

**Symptom:** "Unable to allocate 12.0 GiB for an array with shape (201220, 4000) and data type complex128."

**Root Cause:** Chandrayaan orbital tracks are narrow, extremely long strips (4,000 × 201,220 pixels ≈ 800 million pixels). Computing 2D FFTs with 24 log-Gabor filter orientations on the full array required over 12.8 GB of contiguous RAM per intermediate tensor.

**Resolution:** Implemented `phase_congruency_tiled` in `illumination.py`, breaking large images into overlapping 2,048 × 2,048 blocks. Reduced RAM usage to a constant ~60 MB.

### Challenge B: Gaussian Pyramid CPU Freezing (3–5 min hangs)

**Root Cause:** Full-image loading (`src.read()`) transferred 1.6 GB of raw pixel data into memory and computed multi-octave Gaussian blurs across the entire 200,000-pixel height.

**Resolution:** Implemented windowed reading in `image_loader.py` using `rasterio.windows.Window`. For large tracks, it extracts a central 4,096 × 4,096 patch directly from disk, cutting loading and pyramid construction from minutes to under 0.5 seconds. (Note: subsequently enhanced with georeferenced overlap-aware cropping via `load_overlapping_pair()` to ensure source and reference windows cover the same lunar region.)

---

## 4. Computer Vision & Alignment Pipeline Bugs

### Challenge A: Sub-Pixel Refinement Crash

**Symptom:** "OpenCV(5.0.0) (-215:Assertion failed) (depth == CV_8U || depth == CV_32F) in cv::matchTemplate."

**Root Cause:** OpenCV's Normalized Cross-Correlation (`cv2.matchTemplate`) strictly requires 8-bit (uint8) or 32-bit float (float32) rasters. In `subpixel.py`, inputs were only cast to float if they were uint8. Because lunar data was uint16/int16, uncast 16-bit integer patches reached OpenCV and triggered an assertion failure.

**Resolution:** Updated `subpixel.py` to cast template and search patches to float32 unconditionally.

### Challenge B: Insufficient Tie-Points (Required: 3, got: 0)

**Root Cause:** Arbitrary orbit strips from Chandrayaan and LRO NAC taken months/years apart do not naturally align in their absolute geometric centers without explicit geospatial bounding boxes. Central crops were looking at two completely different parts of the Moon.

**Resolution:** Built `generate_mock_terrain.py` to generate overlapping, crater-textured test rasters with known rotations, translations, and simulated sensor illumination differences. Additionally hardened the loader with `load_overlapping_pair()` which computes the geographic intersection of two georeferenced images and crops both to the shared region.

---

## 5. Frontend Canvas Rendering & Data Exports

### Challenge A: Black Canvas / `net::ERR_FILE_NOT_FOUND` on Blob URLs

**Root Cause:** Web browsers have no native TIFF decoders. Loading raw `.tif` file handles into an HTML Image via `URL.createObjectURL` fails silently.

**Resolution:** Refactored `MatchPointOverlay.tsx` and `App.tsx` to detect TIFF inputs and load backend-rendered PNG previews (`/jobs/{job_id}/preview/source` and `/preview/reference`) instead.

### Challenge B: GeoJSON Opening as Raw Text

**Root Cause:** The `/matches` route returned a raw JSON dict, prompting browsers to display text instead of downloading.

**Resolution:** Changed return type to a `JSONResponse` with header `Content-Disposition: attachment; filename=matches.geojson`.

### Challenge C: GeoJSON 500 Internal Server Error

**Root Cause:** During endpoint refactoring, `result = job_data["result"]` was inadvertently removed, causing a Python NameError.

**Resolution:** Restored `result = job_data.get("result")` with proper guard clauses in `routes.py`.

---

## 6. Cloud & Production Hosting Limits

**Constraints:** Free-tier cloud platforms (Render, Railway, Fly.io, Hugging Face Spaces) enforce 256 MB–512 MB memory caps and lack GPU acceleration. A production build bundling PyTorch, Kornia, SuperPoint, LightGlue, and GDAL/rasterio exceeds these limits.

**Resolution:**

- Deployed the decoupled static React frontend to Vercel (https://lunar-align.vercel.app).
- Kept the computationally heavy PyTorch backend on a local CPU/GPU orchestrator, connected through direct REST APIs.

---

## Summary Table

| Category | Challenge | Impact | Solution |
|----------|-----------|--------|----------|
| Data Format | PDS4/PDS3 `.img` not recognized by GDAL | Cannot ingest real ISRO/NASA data | `convert_pds_to_tiff.py` XML label parser → GeoTIFF + auto `.zip` extraction |
| Network | Starlette multipart size limit | 400 error on files > 1 MB | `LargeUploadMiddleware`, 4 GB parser limit, 8 MB chunked disk streams |
| Network | Proxy connection resets | `ERR_CONNECTION_RESET` on large uploads | Direct Uvicorn routing via `.env`, bypassing Vite/Node proxy |
| Network | CORS wildcard + credentials | Browser blocks responses | `allow_credentials=False` for stateless API |
| Memory | Phase congruency full-array FFT | 12.8 GB OOM crash on 800 MP tracks | `phase_congruency_tiled` overlapping 2,048 blocks → ~60 MB constant |
| Memory | Gaussian pyramid full-image load | 3–5 min CPU freeze | Windowed `rasterio.Window` reads → < 0.5 s |
| CV Pipeline | OpenCV `matchTemplate` dtype assertion | Sub-pixel refinement crash on 16-bit data | Unconditional `float32` casting in `subpixel.py` |
| CV Pipeline | Non-overlapping orbit strips | Zero tie-points (required 3) | Mock terrain generator + georeferenced `load_overlapping_pair()` |
| Frontend | Browser cannot decode TIFF | Black canvas / file-not-found | Backend-rendered PNG preview endpoints |
| Frontend | GeoJSON served as raw text | Opens in browser instead of downloading | `JSONResponse` with `Content-Disposition: attachment` |
| Frontend | Missing `result` binding | GeoJSON 500 NameError | Restored `result = job_data.get("result")` with guards |
| Hosting | Free-tier memory/GPU caps | Build exceeds 256–512 MB limits | Static frontend on Vercel + local PyTorch backend via REST |
