"""FastAPI Main Application for RoadLens AI.

Initializes FastAPI app, configures CORS, mounts API routes,
and provides exception handling.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.app.config import settings
from backend.app.api.routes import router as api_router
from backend.app.api.validation import ValidationError
from backend.app.performance.device import device_manager
from backend.app.vision.detector import road_detector
from backend.app.vision.ocr import ocr_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm vision models during application startup so request telemetry measures pure inference
    road_detector.warmup()
    ocr_service.warmup()
    yield


app = FastAPI(
    title="RoadLens AI API",
    description="AI-Powered Road Perception & Road Intelligence Perception Service",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Validation Exception Handler
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": "ValidationError", "detail": exc.message},
    )

# Include API Router
app.include_router(api_router)


@app.get("/")
async def root():
    return {
        "project": "RoadLens AI",
        "description": "AI-Powered Road Perception & Road Intelligence",
        "version": "0.1.0",
        "status": "online",
        "device": device_manager.device_name,
        "rocm_available": device_manager.is_rocm,
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.host, port=settings.port, reload=True)
