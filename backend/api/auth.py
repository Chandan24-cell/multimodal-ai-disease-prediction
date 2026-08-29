# backend/api/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Any, Dict

from database.mongodb import db, settings
from database.schemas import UserCreate, UserDB, Token, TokenData

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Security utilities
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

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
    """Register a new user (Doctor/Admin)."""
    existing_user = await db.db.users.find_one({"$or": [{"username": user_in.username}, {"email": user_in.email}]})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username or email already registered")
    
    user_dict = user_in.model_dump()
    user_dict["hashed_password"] = get_password_hash(user_dict.pop("password"))
    
    result = await db.db.users.insert_one(user_dict)
    return {"message": "User registered successfully", "user_id": str(result.inserted_id)}

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and return JWT."""
    user = await db.db.users.find_one({"username": form_data.username})
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}