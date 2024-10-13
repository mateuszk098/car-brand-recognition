"""Utility functions for the service layer."""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import bcrypt
import cv2 as cv
import numpy as np
from fastapi import FastAPI
from fastapi.security import OAuth2PasswordBearer
from numpy.typing import NDArray

from resnet import CarClassifier

from ..data import init_db
from ..errors import ImageDecodingError

oauth2_bearer = OAuth2PasswordBearer(tokenUrl="/user/token")
MODEL = os.getenv("MODEL", "SEResNet3")
BACKEND = os.getenv("BACKEND", "PYTORCH")


@asynccontextmanager
async def on_startup(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize the database and deep learning model."""
    init_db()
    CarClassifier(MODEL, BACKEND)
    yield


async def load_model() -> CarClassifier:
    """Load the pre-trained SE-ResNet model."""
    return CarClassifier(MODEL, BACKEND)


def decode_image(content: bytes) -> NDArray:
    """Decode an image from bytes."""
    buffer = np.frombuffer(content, dtype=np.uint8)
    image = cv.imdecode(buffer, cv.IMREAD_COLOR)
    if image is None:
        raise ImageDecodingError()
    return image


def encode_password(password: str) -> bytes:
    """Encode a plain text password into a hashed password."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())


def verify_password(plain: str, hashed: bytes) -> bool:
    """Verify a plain text password against a hashed password."""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed)
