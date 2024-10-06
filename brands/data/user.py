from sqlalchemy.orm import Session

from .models import User


def get_user(username: str, db: Session) -> User | None:
    return db.query(User).filter_by(username=username).first()


def get_users(db: Session) -> list[User]:
    return db.query(User).all()


def create(user: User, db: Session) -> User:
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
