from typing import Generator

from sqlalchemy.orm import Session

from .database import engine, session_factory
from .models import Base


def get_db() -> Generator[Session, None, None]:
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    Base.metadata.create_all(engine)
