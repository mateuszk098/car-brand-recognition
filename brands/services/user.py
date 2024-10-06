import os
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
from dotenv import find_dotenv, load_dotenv
from fastapi import HTTPException, status
from jwt.exceptions import PyJWTError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..data import user as data
from ..data.models import Task, User
from ..errors.user import IncorrectPasswordError, MissingUserError, UserAlreadyExistsError
from ..schemas.user import TaskCreate, TaskSchema, TokenMetadata, UserCreate, UserSchema

load_dotenv(find_dotenv())

SECRET_KEY = os.getenv("SECRET_KEY", "")
ALGORITHM = os.getenv("ALGORITHM", "HS256")


def auth_user(username: str, password: str, db: Session) -> UserSchema:
    user = get_user(username, db)
    if not verify_password(password, user.hashed_password):
        raise IncorrectPasswordError()
    return user


def create_access_token(username: str, user_id: int, user_role: str, expires: timedelta) -> str:
    expiry_time = datetime.now(UTC) + expires
    payload = {
        "sub": username,
        "id": user_id,
        "role": user_role,
        "exp": expiry_time,
    }
    return jwt.encode(payload, SECRET_KEY, ALGORITHM)


def get_token_metadata(token: str) -> TokenMetadata:
    try:
        payload = jwt.decode(token, SECRET_KEY, [ALGORITHM])
    except PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
    else:
        username = payload.get("sub")
        user_id = payload.get("id")
        user_role = payload.get("role")
        if any(x is None for x in (username, user_id, user_role)):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
        return TokenMetadata(username=username, user_id=user_id, user_role=user_role)


def get_user(username: str, db: Session) -> UserSchema:
    db_user = data.get_user(username, db)
    if db_user is None:
        raise MissingUserError(username)
    return UserSchema.model_validate(db_user)


def get_users(db: Session) -> list[UserSchema]:
    db_users = data.get_users(db)
    return list(UserSchema.model_validate(db_user) for db_user in db_users)


def create_user(user: UserCreate, db: Session) -> UserSchema:
    db_user = User(
        username=user.username,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        role=user.role,
        hashed_password=encode_password(user.password),
    )
    try:
        db_user = data.create_user(db_user, db)
    except IntegrityError:
        raise UserAlreadyExistsError(user.username)
    else:
        return UserSchema.model_validate(db_user)


def create_task(task: TaskCreate, user_id: int, db: Session) -> TaskSchema:
    db_task = Task(
        user_id=user_id,
        prediction=task.prediction,
        probability=task.probability,
    )
    data.create_task(db_task, db)
    return TaskSchema.model_validate(db_task)


def get_tasks(user_id: int, db: Session) -> list[TaskSchema]:
    db_tasks = data.get_tasks(user_id, db)
    return list(TaskSchema.model_validate(db_task) for db_task in db_tasks)


def encode_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())


def verify_password(plain: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed)
