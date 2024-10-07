import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from resnet import ArchType, SEResNet, load_se_resnet, predict

from ..errors.classifier import ImageDecodingError


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
