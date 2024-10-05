import os

from dotenv import find_dotenv, load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv(find_dotenv())

APP_DB_URL = os.getenv("APP_DB_URL", "sqlite:///./brands.db")

engine = create_engine(APP_DB_URL, connect_args={"check_same_thread": False})
session_factory = sessionmaker(engine, autoflush=False, autocommit=False)
