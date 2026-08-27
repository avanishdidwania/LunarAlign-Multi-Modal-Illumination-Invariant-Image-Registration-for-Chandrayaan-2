FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies (required for OpenCV and GDAL dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging configuration and codebase
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the lunar-reg package with its dependencies
RUN pip install --no-cache-dir .

# Expose backend API port
EXPOSE 8000

# Copy startup manager script
COPY entrypoint.py ./

ENTRYPOINT ["python", "entrypoint.py"]
