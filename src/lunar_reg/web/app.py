from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from lunar_reg.web.routes import router

# Patch Starlette's MultiPartParser to allow large lunar .img file uploads (up to 4 GB)
try:
    from starlette.formparsers import MultiPartParser
    MultiPartParser.max_file_size = 4 * 1024 * 1024 * 1024  # 4 GB
except Exception:
    pass

try:
    import multipart.multipart as _mp
    _mp.MAX_BODY_SIZE = 4 * 1024 * 1024 * 1024
    _mp.MAX_FILE_SIZE = 4 * 1024 * 1024 * 1024
except Exception:
    pass

app = FastAPI(
    title="Lunar Image Registration Pipeline API",
    description="API for high-precision satellite image registration on lunar terrain (SIH 26166).",
    version="0.1.0"
)

# Allow very large file uploads (up to 4 GB) for full-resolution lunar .img files
class LargeUploadMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Override Starlette's default 1 MB multipart form max size
        request._max_upload_size = 4 * 1024 * 1024 * 1024  # 4 GB
        return await call_next(request)

app.add_middleware(LargeUploadMiddleware)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes router
app.include_router(router)
