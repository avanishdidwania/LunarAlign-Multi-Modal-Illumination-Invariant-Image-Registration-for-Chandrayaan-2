from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from lunar_reg.web.routes import router

app = FastAPI(
    title="Lunar Image Registration Pipeline API",
    description="API for high-precision satellite image registration on lunar terrain (SIH 26166).",
    version="0.1.0"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes router
app.include_router(router)
