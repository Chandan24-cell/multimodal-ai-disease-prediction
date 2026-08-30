# backend/main.py
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging

# Existing backend modules use top-level imports (api, database, inference).
# Add this directory so `backend.main` works when launched from the repository root.
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from database.mongodb import connect_to_mongo, close_mongo_connection, db, settings
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

# Resolve from this file, never from the shell's current working directory.
FRONTEND_DIST_DIR = Path(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown events."""
    # Startup
    logger.info("Starting up application...")
    try:
        await connect_to_mongo()
    except Exception:
        logger.exception("Application started with MongoDB unavailable")
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

    configured_origins = {
        "http://localhost",
        "http://localhost:5173",
        "http://localhost:3000",
        settings.FRONTEND_URL.rstrip("/"),
    }
    allowed_origins = [origin for origin in configured_origins if origin]

    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
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
        database_status = "connected" if db.client is not None and db.db is not None else "disconnected"
        return {
            "status": "healthy" if database_status == "connected" else "degraded",
            "database": database_status,
            "disclaimer": "Research prototype only. Not for clinical diagnosis.",
        }

    # A combined deployment can serve the Vite build from FastAPI. In the
    # normal Compose deployment Nginx serves it from the separate frontend
    # container, so this mount is intentionally conditional.
    if FRONTEND_DIST_DIR.is_dir():
        logger.info("Serving frontend build from %s", FRONTEND_DIST_DIR)
        assets_dir = FRONTEND_DIST_DIR / "assets"
        if assets_dir.is_dir():
            application.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        @application.get("/{full_path:path}", include_in_schema=False)
        async def serve_frontend(full_path: str):
            """Serve assets and index.html for React client-side routes."""
            if full_path.startswith("api/"):
                return JSONResponse(status_code=404, content={"detail": "API endpoint not found"})
            requested_file = FRONTEND_DIST_DIR / full_path
            if full_path and requested_file.is_file():
                return FileResponse(requested_file)
            return FileResponse(FRONTEND_DIST_DIR / "index.html")
    else:
        logger.info("Frontend build not found at %s; expecting a separate frontend service", FRONTEND_DIST_DIR)

        @application.get("/{full_path:path}", include_in_schema=False)
        async def frontend_build_missing(full_path: str):
            if full_path.startswith("api/"):
                return JSONResponse(status_code=404, content={"detail": "API endpoint not found"})
            return {"detail": "Frontend build not found"}

    return application

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
