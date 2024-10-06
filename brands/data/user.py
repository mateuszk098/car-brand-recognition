from sqlalchemy.orm import Session

from .models import Task, User


def create_user(user: User, db: Session) -> User:
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user(username: str, db: Session) -> User | None:
    return db.query(User).filter_by(username=username).first()


def get_users(db: Session) -> list[User]:
    return db.query(User).all()


def create_task(task: Task, db: Session) -> Task:
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def get_tasks(user_id: int, db: Session) -> list[Task]:
    return db.query(Task).filter_by(user_id=user_id).all()
