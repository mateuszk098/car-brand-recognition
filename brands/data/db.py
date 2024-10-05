import os
from typing import Generator

from dotenv import find_dotenv, load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

load_dotenv(find_dotenv())

APP_DB_URL = os.getenv("APP_DB_URL", "sqlite:///./brands.db")

engine = create_engine(APP_DB_URL, connect_args={"check_same_thread": False})
session_factory = sessionmaker(engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    db = session_factory()
    try:
        yield db
    finally:
        db.close()
