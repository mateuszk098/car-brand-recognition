from fastapi import UploadFile
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.errors import (
    ImageDecodingError,
    ImageDecodingHTTPError,
    RecordNotFoundError,
    UserAlreadyExistsError,
    UserNotFoundError,
)
from resnet import SEResNet

from ..data.models import Task, User
from ..data.repositories import TaskRepository, UserRepository
from ..schema.user import TaskSchema, UserCreate, UserSchema
from .utils import decode_image, encode_password, predict_brand


def get_user_by_username(username: str, db: Session) -> UserSchema:
    """Get a user by username."""
    try:
        user = UserRepository.get_user_by_username(username, db)
    except RecordNotFoundError:
        raise UserNotFoundError(username)
    return UserSchema.model_validate(user)


def get_users(db: Session) -> list[UserSchema]:
    """Get all the existing users."""
    users = UserRepository.get_users(db)
    return list(UserSchema.model_validate(user) for user in users)


def create_user(user: UserCreate, db: Session) -> UserSchema:
    """Create a new user in database."""
    new_user = User(
        username=user.username,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        role=user.role,
        hashed_password=encode_password(user.password.get_secret_value()),
    )
    try:
        new_user = UserRepository.create_user(new_user, db)
    except IntegrityError:
        raise UserAlreadyExistsError(user.username)
    else:
        return UserSchema.model_validate(new_user)


def get_tasks_for_user(user_id: int, db: Session) -> list[TaskSchema]:
    """Get all tasks for a user."""
    tasks = TaskRepository.get_tasks_for_user(user_id, db)
    return list(TaskSchema.model_validate(task) for task in tasks)


async def create_task(upload: UploadFile, topk: int, user_id: int, model: SEResNet, db: Session) -> TaskSchema:
    """Create a deep learning model's prediction task for a given user."""
    try:
        file_content = await upload.read()
        image = decode_image(file_content)
    except ImageDecodingError:
        raise ImageDecodingHTTPError()
    else:
        prediction = await predict_brand(image, model, topk)
        brands, probs = zip(*prediction)
        task = Task(
            user_id=user_id,
            name=upload.filename,
            content=upload.content_type,
            brands=", ".join(brands),
            probs=", ".join(str(p) for p in probs),
        )
        TaskRepository.create_task(task, db)
        return TaskSchema.model_validate(task)
    finally:
        upload.file.close()
