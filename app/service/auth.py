"""Authentication via JWT."""

import os
from datetime import UTC, datetime, timedelta

import jwt
from dotenv import find_dotenv, load_dotenv
from fastapi import Depends, HTTPException, status
from jwt.exceptions import PyJWTError
from sqlalchemy.orm import Session

from ..data import get_db
from ..errors import InvalidCredentialsError, InvalidPasswordError
from ..schema.user import UserSchema
from .user import get_user_by_username
from .utils import oauth2_bearer, verify_password

load_dotenv(find_dotenv())

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")


def authenticate_user(username: str, password: str, db: Session) -> UserSchema:
    """Authenticate a user with a username and password."""
    user = get_user_by_username(username, db)
    if not verify_password(password, user.hashed_password):
        raise InvalidPasswordError()
    return user


def create_access_token(username: str, expires: timedelta) -> str:
    """Create a JWT access token."""
    expiry_time = datetime.now(UTC) + expires
    payload = {"sub": username, "exp": expiry_time}
    return jwt.encode(payload, JWT_SECRET_KEY, JWT_ALGORITHM)


def extract_username_from_token(token: str) -> str:
    """Extract the username from a JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, [JWT_ALGORITHM])
    except PyJWTError:
        raise InvalidCredentialsError()
    else:
        username: str = payload.get("sub")
        if username is None:
            raise InvalidCredentialsError()
        return username


async def get_current_user(token: str = Depends(oauth2_bearer), db: Session = Depends(get_db)) -> UserSchema:
    """Get the current user from a JWT token."""
    try:
        username = extract_username_from_token(token)
    except InvalidCredentialsError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=e.detail)
    else:
        return get_user_by_username(username, db)
