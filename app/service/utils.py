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
    """
    Initialize the database and deep learning model on application startup.

    This asynchronous generator function is intended to be used as a startup event handler
    for a FastAPI application. It performs the following tasks:
    1. Initializes the database by calling the `init_db` function.
    2. Instantiates the `CarClassifier` with the specified `MODEL` and `BACKEND`.

    Args:
        app (FastAPI): The FastAPI application instance.
    Yields:
        None: This function is an asynchronous generator and will yield once after initialization.
    """
    init_db()
    CarClassifier(MODEL, BACKEND)
    yield


async def load_model() -> CarClassifier:
    """
    Load the pre-trained SE-ResNet model.
    Returns:
        CarClassifier: An instance of the CarClassifier with the pre-trained model loaded.
    """
    return CarClassifier(MODEL, BACKEND)


def decode_image(content: bytes) -> NDArray:
    """
    Decode an image from bytes.
    Args:
        content (bytes): The byte content of the image to decode.
    Returns:
        NDArray: The decoded image as a NumPy array.
    Raises:
        ImageDecodingError: If the image cannot be decoded.
    """
    buffer = np.frombuffer(content, dtype=np.uint8)
    image = cv.imdecode(buffer, cv.IMREAD_COLOR)
    if image is None:
        raise ImageDecodingError()
    return image


def encode_password(password: str) -> bytes:
    """
    Encode a plain text password into a hashed password.
    Args:
        password (str): The plain text password to be encoded.
    Returns:
        bytes: The hashed password.
    """
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())


def verify_password(plain: str, hashed: bytes) -> bool:
    """
    Verify a plain text password against a hashed password.
    Args:
        plain (str): The plain text password to verify.
        hashed (bytes): The hashed password to compare against.
    Returns:
        bool: True if the plain text password matches the hashed password, False otherwise.
    """
    return bcrypt.checkpw(plain.encode("utf-8"), hashed)
