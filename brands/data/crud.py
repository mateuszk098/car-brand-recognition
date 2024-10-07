from sqlalchemy.orm import Session

from .models import Task, User


def create_user(user: User, db: Session) -> User:
    """Create a new user in database."""
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user(username: str, db: Session) -> User | None:
    """Get a user by username."""
    return db.query(User).filter_by(username=username).first()


def get_users(db: Session) -> list[User]:
    """Get all users from database."""
    return db.query(User).all()


def create_user_task(task: Task, db: Session) -> Task:
    """Create a new task in database."""
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def get_user_tasks(user_id: int, db: Session) -> list[Task]:
    """Get all tasks of a user with given ID."""
    return db.query(Task).filter_by(user_id=user_id).all()
