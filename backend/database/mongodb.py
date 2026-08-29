# backend/database/mongodb.py
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic_settings import BaseSettings
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Application settings loaded from .env file."""
    APP_ENV: str = "development"
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "multimodal_healthcare"
    JWT_SECRET_KEY: str = "super_secret_key_change_in_production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

class Database:
    """Singleton-like class to manage MongoDB connection."""
    client: Optional[AsyncIOMotorClient] = None
    db: Optional[object] = None

db = Database()

async def connect_to_mongo():
    """Establishes connection to MongoDB."""
    logger.info("Connecting to MongoDB...")
    db.client = AsyncIOMotorClient(settings.MONGODB_URI)
    db.db = db.client[settings.MONGODB_DB_NAME]
    
    # Create indexes for performance
    await db.db.patients.create_index("email", unique=True)
    await db.db.users.create_index("username", unique=True)
    logger.info("Connected to MongoDB successfully.")

async def close_mongo_connection():
    """Closes the MongoDB connection."""
    if db.client:
        logger.info("Closing MongoDB connection...")
        db.client.close()
        logger.info("MongoDB connection closed.")