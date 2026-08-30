# backend/api/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Any, Dict
import logging

from database.mongodb import db, settings
from database.schemas import UserCreate, UserDB, Token, TokenData

router = APIRouter(prefix="/auth", tags=["Authentication"])
logger = logging.getLogger(__name__)

# Security utilities
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserDB:
    """Dependency to extract and validate the current user from the JWT."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = await db.db.users.find_one({"username": token_data.username})
    if user is None:
        raise credentials_exception
    
    # Convert MongoDB _id to string for Pydantic
    user["_id"] = str(user["_id"])
    return UserDB(**user)

@router.post("/register", response_model=Dict[str, str], status_code=status.HTTP_201_CREATED)
async def register(user_in: UserCreate):
    """
    Register a new user with role-based access control.
    - First user registered in the system gets role="Admin"
    - All subsequent users get role="Staff"
    """
    if db.db is None:
        logger.error("Registration rejected because MongoDB is not connected")
        raise HTTPException(status_code=503, detail="Authentication service unavailable")

    existing_user = await db.db.users.find_one({"$or": [{"username": user_in.username}, {"email": user_in.email}]})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username or email already registered")
    
    user_dict = user_in.model_dump()
    user_dict["hashed_password"] = get_password_hash(user_dict.pop("password"))
    
    # Determine if this is the first user in the system. Never allow a later
    # public registration to grant itself Admin access.
    user_count = await db.db.users.count_documents({})
    user_dict["role"] = "Admin" if user_count == 0 else "Staff"
    
    result = await db.db.users.insert_one(user_dict)
    return {
        "message": "User registered successfully",
        "user_id": str(result.inserted_id),
        "role": user_dict["role"]
    }

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and return JWT."""
    logger.info("Login attempt username=%s", form_data.username)
    if db.db is None:
        logger.error("Login rejected because MongoDB is not connected")
        raise HTTPException(status_code=503, detail="Authentication service unavailable")

    user = await db.db.users.find_one({"username": form_data.username})
    password_valid = bool(user and verify_password(form_data.password, user["hashed_password"]))
    if not user:
        logger.warning("Login failed: unknown username=%s", form_data.username)
    elif not password_valid:
        logger.warning("Login failed: invalid password username=%s", form_data.username)

    if not password_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    logger.info("Login succeeded username=%s role=%s", user["username"], user.get("role", "Staff"))
    access_token = create_access_token(data={"sub": user["username"], "role": user.get("role", "Staff")})
    return {"access_token": access_token, "token_type": "bearer"}
