"""Custom exceptions for the classifier module."""


class ImageDecodingError(Exception):
    """Raised when an image cannot be decoded."""

    def __init__(self, message="Cannot decode image") -> None:
        self.message = message
        super().__init__(self.message)
