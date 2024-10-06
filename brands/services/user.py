import bcrypt
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..data import user as data
from ..data.models import Task, User
from ..errors.user import MissingUserError, UserAlreadyExistsError
from ..schemas.user import TaskCreate, TaskSchema, UserCreate, UserSchema


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
