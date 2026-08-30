# backend/database/mongodb.py
from typing import Optional
from pathlib import Path
import logging

from motor.motor_asyncio import AsyncIOMotorClient
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from the backend .env file."""
    APP_ENV: str = "development"
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "multimodal_healthcare"
    JWT_SECRET_KEY: str = "super_secret_key_change_in_production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    FRONTEND_URL: str = "http://localhost:5173"
    LLM_PROVIDER: str = "mock"
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL: Optional[str] = None

    class Config:
        env_file = Path(__file__).resolve().parent.parent / ".env"
        extra = "ignore"


settings = Settings()


class Database:
    """Singleton-like class to manage MongoDB connection."""
    client: Optional[AsyncIOMotorClient] = None
    db: Optional[object] = None


db = Database()

async def connect_to_mongo():
    """Establishes connection to MongoDB."""
    logger.info("Connecting to MongoDB database=%s", settings.MONGODB_DB_NAME)
    try:
        db.client = AsyncIOMotorClient(settings.MONGODB_URI, serverSelectionTimeoutMS=5000)
        db.db = db.client[settings.MONGODB_DB_NAME]
        await db.client.admin.command("ping")
        await db.db.patients.create_index("email", unique=True)
        await db.db.users.create_index("username", unique=True)
        logger.info("Connected to MongoDB successfully.")
    except Exception:
        logger.exception("MongoDB connection failed")
        if db.client:
            db.client.close()
        db.client = None
        db.db = None
        raise

async def close_mongo_connection():
    """Closes the MongoDB connection."""
    if db.client:
        logger.info("Closing MongoDB connection...")
        db.client.close()
        logger.info("MongoDB connection closed.")
