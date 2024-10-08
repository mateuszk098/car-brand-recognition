from contextlib import asynccontextmanager
from typing import AsyncGenerator

import bcrypt
import cv2 as cv
import numpy as np
from fastapi import FastAPI
from fastapi.security import OAuth2PasswordBearer
from numpy.typing import NDArray

from app.errors import ImageDecodingError
from resnet import ArchType, SEResNet, download_pretrained_weights, load_se_resnet, predict

oauth2_bearer = OAuth2PasswordBearer(tokenUrl="/user/token")


@asynccontextmanager
async def download_model_on_startup(app: FastAPI) -> AsyncGenerator[None, None]:
    """Download pre-trained weights for SE-ResNet model."""
    download_pretrained_weights(ArchType.SEResNet3)
    yield


async def load_model() -> SEResNet:
    """Load the pre-trained SE-ResNet model."""
    return load_se_resnet(ArchType.SEResNet3)


async def predict_brand(image: NDArray, model: SEResNet, topk: int = 5) -> list[tuple[str, float]]:
    """Predict the brand of a car from an image."""
    return predict(image, model, topk)


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
