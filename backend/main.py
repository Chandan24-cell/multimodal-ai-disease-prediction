# backend/main.py
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import sys

from database.mongodb import connect_to_mongo, close_mongo_connection
from api.auth import router as auth_router
from api.patients import router as patients_router
from api.prediction import router as prediction_router
from api.reports import router as reports_router
from api.rag import router as rag_router
from api.feedback import router as feedback_router

# Structured logging for audit trails
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(module)s", "message": "%(message)s"}',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# The frontend build is created by Render's build command at ``frontend/dist``.
FRONTEND_DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown events."""
    # Startup
    logger.info("Starting up application...")
    await connect_to_mongo()
    yield
    # Shutdown
    logger.info("Shutting down application...")
    await close_mongo_connection()

def create_app() -> FastAPI:
    """Application factory."""
    application = FastAPI(
        title="Multimodal Healthcare AI API",
        description="API for Multimodal AI-Based Intelligent Healthcare System. ⚠️ FOR RESEARCH AND DECISION SUPPORT ONLY. NOT FOR CLINICAL DIAGNOSIS.",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan
    )

    # CORS Middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost", "http://localhost:5173", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include Routers
    application.include_router(auth_router, prefix="/api")
    application.include_router(patients_router, prefix="/api")
    application.include_router(prediction_router, prefix="/api")
    application.include_router(reports_router, prefix="/api")
    application.include_router(rag_router, prefix="/api")
    application.include_router(feedback_router, prefix="/api")

    @application.get("/api/health", tags=["Health"])
    async def health_check():
        return {"status": "healthy", "disclaimer": "Research prototype only. Not for clinical diagnosis."}

    # Mount only the hashed Vite assets.  The catch-all below then returns
    # index.html for client-side React routes such as /dashboard.
    assets_dir = FRONTEND_DIST_DIR / "assets"
    if assets_dir.is_dir():
        application.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @application.get("/{full_path:path}", include_in_schema=False)
    async def serve_frontend(full_path: str):
        """Serve the compiled React app and support client-side routing."""
        if full_path.startswith("api/"):
            return JSONResponse(status_code=404, content={"detail": "API endpoint not found"})

        if not FRONTEND_DIST_DIR.is_dir():
            return {"detail": "Frontend build not found"}

        requested_file = FRONTEND_DIST_DIR / full_path
        if full_path and requested_file.is_file():
            return FileResponse(requested_file)
        return FileResponse(FRONTEND_DIST_DIR / "index.html")

    return application

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
