class ImageDecodingError(Exception):
    def __init__(self, message="Cannot decode image") -> None:
        self.message = message
        super().__init__(self.message)
