"""Database configuration and session factory."""

import os

from dotenv import find_dotenv, load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv(find_dotenv())

APP_DB_URL = os.getenv("APP_DB_URL", "sqlite:///./brands.db")
if APP_DB_URL.startswith("sqlite"):
    APP_DB_URL += "?check_same_thread=False"

engine = create_engine(APP_DB_URL)
session_factory = sessionmaker(engine, autoflush=False, autocommit=False)
