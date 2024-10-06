from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..data import user as data
from ..data.models import User
from ..errors.user import MissingUserError, UserAlreadyExistsError
from ..schemas.user import UserCreate, UserSchema


def get_user(username: str, db: Session) -> UserSchema:
    db_user = data.get_user(username, db)
    if db_user is None:
        raise MissingUserError(username)
    return UserSchema.model_validate(db_user)


def get_users(db: Session) -> list[UserSchema]:
    db_users = data.get_users(db)
    return list(UserSchema.model_validate(db_user) for db_user in db_users)


def create(user: UserCreate, db: Session) -> UserSchema:
    fake_hash_password = b"fakehashedpassword"
    db_user = User(username=user.username, email=user.email, hashed_password=fake_hash_password)
    try:
        db_user = data.create(db_user, db)
    except IntegrityError:
        raise UserAlreadyExistsError(user.username)
    else:
        return UserSchema.model_validate(db_user)
