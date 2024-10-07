import os
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
from dotenv import find_dotenv, load_dotenv
from fastapi import HTTPException, UploadFile, status
from jwt.exceptions import PyJWTError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from resnet import SEResNet

from ..data import crud
from ..data.models import Task, User
from ..errors.classifier import ImageDecodingError
from ..errors.user import IncorrectPasswordError, MissingUserError, UserAlreadyExistsError
from ..schemas.user import TaskSchema, UserCreate, UserSchema
from .classifier import decode_image, predict_brand

load_dotenv(find_dotenv())

SECRET_KEY = os.getenv("SECRET_KEY", "")
ALGORITHM = os.getenv("ALGORITHM", "HS256")


def create_user(user: UserCreate, db: Session) -> UserSchema:
    """Create a new user in database."""
    db_user = User(
        username=user.username,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        role=user.role,
        hashed_password=_encode_password(user.password.get_secret_value()),
    )
    try:
        db_user = crud.create_user(db_user, db)
    except IntegrityError:
        raise UserAlreadyExistsError(user.username)
    else:
        return UserSchema.model_validate(db_user)


def get_user(username: str, db: Session) -> UserSchema:
    """Get a user by username."""
    db_user = crud.get_user(username, db)
    if db_user is None:
        raise MissingUserError(username)
    return UserSchema.model_validate(db_user)


def get_users(db: Session) -> list[UserSchema]:
    """Get all the existing users."""
    db_users = crud.get_users(db)
    return list(UserSchema.model_validate(db_user) for db_user in db_users)


async def create_task(upload: UploadFile, user_id: int, db: Session, model: SEResNet, topk: int = 5) -> TaskSchema:
    """Create a deep learning task for a given user."""
    try:
        content = await upload.read()
        image = decode_image(content)
    except ImageDecodingError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e))
    else:
        prediction = await predict_brand(image, model, topk)
        brands, probs = zip(*prediction)
        db_task = Task(
            user_id=user_id,
            name=upload.filename,
            content=upload.content_type,
            brands=", ".join(brands),
            probs=", ".join(str(p) for p in probs),
        )
        crud.create_user_task(db_task, db)
        return TaskSchema.model_validate(db_task)
    finally:
        upload.file.close()


def get_user_tasks(user_id: int, db: Session) -> list[TaskSchema]:
    """Get all tasks for a user."""
    db_tasks = crud.get_user_tasks(user_id, db)
    return list(TaskSchema.model_validate(db_task) for db_task in db_tasks)


def authenticate_user(username: str, password: str, db: Session) -> UserSchema:
    """Authenticate a user with a username and password."""
    user = get_user(username, db)
    if not _verify_password(password, user.hashed_password):
        raise IncorrectPasswordError()
    return user


def create_access_token(username: str, expires: timedelta) -> str:
    """Create a JWT access token."""
    expiry_time = datetime.now(UTC) + expires
    payload = {"sub": username, "exp": expiry_time}
    return jwt.encode(payload, SECRET_KEY, ALGORITHM)


def extract_username_from_token(token: str) -> str:
    """Extract the username from a JWT token."""
    InvalidCredentials = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, [ALGORITHM])
    except PyJWTError:
        raise InvalidCredentials
    else:
        username: str = payload.get("sub")
        if username is None:
            raise InvalidCredentials
        return username


def _encode_password(password: str) -> bytes:
    """Encode a plain text password into a hashed password."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())


def _verify_password(plain: str, hashed: bytes) -> bool:
    """Verify a plain text password against a hashed password."""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed)
