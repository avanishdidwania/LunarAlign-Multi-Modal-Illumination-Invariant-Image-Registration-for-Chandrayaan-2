# Pretrained Model Weights

This directory provides information and caching configurations for the deep learning models used in the lunar image registration pipeline: **SuperPoint**, **LightGlue**, and **LoFTR**.

## Automatic Download

By default, when running on an internet-enabled environment, PyTorch and Kornia will automatically download the required model weights on their first invocation and cache them to:
- **Linux/Docker**: `~/.cache/torch/hub/checkpoints/`
- **Windows**: `C:\Users\<username>\.cache\torch\hub\checkpoints\`

---

## Offline Deployment (Manual Download)

For air-gapped systems or environments with restricted internet access, you can download the weights manually and copy them to the PyTorch hub checkpoints cache folder before running the pipeline or container.

### 1. SuperPoint Weights
- **Model**: SuperPoint Keypoint Detector
- **File**: `superpoint_v1.pth`
- **Download URL**: [https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth](https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth)

### 2. LightGlue Weights
- **Model**: LightGlue Matcher (SuperPoint variant)
- **File**: `superpoint_lightglue.pth`
- **Download URL**: [https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth](https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth)

### 3. LoFTR Weights
- **Model**: LoFTR Outdoor Matching Model
- **File**: `loftr_outdoor.ckpt`
- **Download URL**: [https://github.com/zju3dv/LoFTR/releases/download/v1.0/loftr_outdoor.ckpt](https://github.com/zju3dv/LoFTR/releases/download/v1.0/loftr_outdoor.ckpt)

---

## Mount Mapping in Docker

When using the provided `docker-compose.yml`, the local `models` directory is mounted to `/app/models`. If you prefer to store weight files inside this directory to share them with the backend container, you can update your code to load weights from `/app/models/` or copy them into the Docker user's `.cache` folder.
