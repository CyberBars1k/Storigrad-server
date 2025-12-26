from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from os import getenv

from .db import get_db
from . import models


SECRET_KEY = getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY is not set")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 168


class RegisterRequest(BaseModel):
    email: EmailStr
    username: str
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    message: str
    token: str


def _hash_password_sha256(password: str) -> str:
    import hashlib

    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def hash_password(password: str) -> str:
    """Hash password using bcrypt (new default)."""
    # passlib provides safe salts + constant-time verification
    from passlib.context import CryptContext

    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return pwd_context.hash(password)


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored hash.

    Supports:
    - bcrypt hashes (new)
    - legacy sha256 hex digests (old)
    """
    if not stored_hash:
        return False

    # bcrypt hashes usually start with "$2" (e.g., $2b$...)
    if stored_hash.startswith("$2"):
        from passlib.context import CryptContext

        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        try:
            return pwd_context.verify(password, stored_hash)
        except Exception:
            return False

    # Legacy sha256 (development-only) support
    return _hash_password_sha256(password) == stored_hash


def verify_and_upgrade_password(db: Session, user: models.User, password: str) -> bool:
    """Verify password and, if the user is on legacy sha256, upgrade to bcrypt.

    Returns True if password is correct, otherwise False.
    """
    if not user or not getattr(user, "password_hash", None):
        return False

    ok = verify_password(password, user.password_hash)
    if not ok:
        return False

    # If legacy sha256, upgrade to bcrypt immediately
    if not user.password_hash.startswith("$2"):
        user.password_hash = hash_password(password)
        db.add(user)
        db.commit()
        db.refresh(user)

    return True


def create_jwt(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {"user_id": user_id, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> models.User:
    """
    Decode JWT from Authorization: Bearer <token> header and return the current user.
    Raises 401 if token is invalid/expired or user not found.
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: Optional[int] = payload.get("user_id")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload.",
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )

    user = db.query(models.User).get(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
        )
    return user