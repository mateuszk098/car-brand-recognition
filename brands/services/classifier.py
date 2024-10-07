import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from resnet.inference import load_se_resnet, predict
from resnet.network.arch import ArchType, SEResNet

from ..errors.classifier import ImageDecodingError


async def load_model() -> SEResNet:
    return load_se_resnet(ArchType.SEResNet3)


async def predict_brand(image: NDArray, model: SEResNet, topk: int = 5) -> list[dict[str, float]]:
    return predict(image, model, topk)


def decode_image(content: bytes) -> NDArray:
    buffer = np.frombuffer(content, dtype=np.uint8)
    image = cv.imdecode(buffer, cv.IMREAD_COLOR)
    if image is None:
        raise ImageDecodingError()
    return image
