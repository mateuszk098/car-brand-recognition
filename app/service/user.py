"""User service module."""

from fastapi import UploadFile
from sqlalchemy.orm import Session

from resnet import CarClassifier

from .. import errors
from ..data.models import Task, User
from ..data.repositories import TaskRepository, UserRepository
from ..schema.user import Password, TaskSchema, UserCreate, UserSchema
from ..service.utils import verify_password
from .utils import decode_image, encode_password


def get_user_by_username(username: str, db: Session) -> UserSchema:
    """
    Retrieve a user by their username.
    Args:
        username (str): The username of the user to retrieve.
        db (Session): The database session to use for the query.
    Returns:
        UserSchema: The schema representing the retrieved user.
    Raises:
        UserNotFoundError: If no user with the given username is found.
    """
    try:
        user = UserRepository.get_user_by_username(username, db)
    except errors.RecordNotFoundError:
        raise errors.UserNotFoundError(username)
    else:
        return UserSchema.model_validate(user)


def get_users(db: Session) -> list[UserSchema]:
    """
    Retrieve all existing users from the database.
    Args:
        db (Session): The database session used to query users.
    Returns:
        list[UserSchema]: A list of user schemas representing the users.
    """
    users = UserRepository.get_users(db)
    return list(UserSchema.model_validate(user) for user in users)


def create_user(user: UserCreate, db: Session) -> UserSchema:
    """
    Creates a new user in the database.
    Args:
        user (UserCreate): An object containing the user details to be created.
        db (Session): The database session to use for the operation.
    Returns:
        UserSchema: The created user object validated against the UserSchema.
    Raises:
        PasswordMismatchError: If the provided password and confirm password do not match.
        UserAlreadyExistsError: If a user with the provided username already exists.
        EmailAlreadyExistsError: If a user with the provided email already exists.
    """
    password = user.password.get_secret_value()
    confirm_password = user.confirm_password.get_secret_value()

    if password != confirm_password:
        raise errors.PasswordMismatchError()

    try:
        existing_user = UserRepository.get_user_by_username(user.username, db)
    except errors.RecordNotFoundError:
        pass
    else:
        raise errors.UserAlreadyExistsError(existing_user.username)

    try:
        existing_user = UserRepository.get_user_by_email(user.email, db)
    except errors.RecordNotFoundError:
        pass
    else:
        raise errors.EmailAlreadyExistsError(existing_user.email)

    new_user = User(
        username=user.username,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        role=user.role,
        hashed_password=encode_password(password),
    )

    new_user = UserRepository.create_user(new_user, db)
    return UserSchema.model_validate(new_user)


def delete_user_by_username(username: str, admin_password: Password, admin_user: UserSchema, db: Session) -> None:
    """
    Deletes a user by their username after verifying the admin's password.
    Args:
        username (str): The username of the user to be deleted.
        admin_password (Password): The admin's password object containing the password and its confirmation.
        admin_user (UserSchema): The admin user's schema containing the hashed password.
        db (Session): The database session to use for the operation.
    Raises:
        PasswordMismatchError: If the admin's password and confirmation password do not match.
        InvalidPasswordError: If the admin's password is invalid.
        UserNotFoundError: If the user with the given username is not found.
    """
    password = admin_password.password.get_secret_value()
    confirm_password = admin_password.confirm_password.get_secret_value()

    if password != confirm_password:
        raise errors.PasswordMismatchError()

    if not verify_password(password, admin_user.hashed_password):
        raise errors.InvalidPasswordError()

    try:
        user = UserRepository.get_user_by_username(username, db)
    except errors.RecordNotFoundError:
        raise errors.UserNotFoundError(username)
    else:
        UserRepository.delete_user(user, db)


def get_tasks_for_user(user_id: int, db: Session) -> list[TaskSchema]:
    """
    Retrieve tasks for a specific user from the database.
    Args:
        user_id (int): The ID of the user whose tasks are to be retrieved.
        db (Session): The database session used to query the tasks.
    Returns:
        list[TaskSchema]: A list of TaskSchema objects representing the user's tasks.
    """
    tasks = TaskRepository.get_tasks_for_user(user_id, db)
    return list(TaskSchema.model_validate(task) for task in tasks)


async def create_task(upload: UploadFile, topk: int, user_id: int, model: CarClassifier, db: Session) -> TaskSchema:
    """
    Creates a new task for car brand recognition.
    Args:
        upload (UploadFile): The uploaded file containing the image to be processed.
        topk (int): The number of top predictions to return.
        user_id (int): The ID of the user creating the task.
        model (CarClassifier): The car classifier model used for making predictions.
        db (Session): The database session for performing database operations.
    Returns:
        TaskSchema: The schema representing the created task.
    Raises:
        ImageDecodingHTTPError: If there is an error decoding the image.
    """
    try:
        file_content = await upload.read()
        image = decode_image(file_content)
        prediction = model(image, topk)
        brands, probs = zip(*prediction)
        task = Task(
            user_id=user_id,
            name=upload.filename,
            content=upload.content_type,
            brands=str(brands),
            probs=str(probs),
        )
        TaskRepository.create_task(task, db)
        return TaskSchema.model_validate(task)
    finally:
        upload.file.close()
