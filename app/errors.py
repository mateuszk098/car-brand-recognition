from fastapi import HTTPException, status


class AppError(Exception):
    """Base class for all application-specific errors."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(self.detail)


# 1. Data Layer Errors
class DatabaseError(AppError):
    """Raised when there is a database operation failure."""

    def __init__(self, detail="Database operation failed") -> None:
        super().__init__(detail)


class RecordNotFoundError(AppError):
    """Raised when a record is not found in the database."""

    def __init__(self, entity_name: str) -> None:
        super().__init__(f"{entity_name} not found.")


# 2. Service Layer Errors
class UserNotFoundError(AppError):
    """Raised when a user is not found in database."""

    def __init__(self, username: str) -> None:
        super().__init__(f"User {username} not found.")


class UserAlreadyExistsError(AppError):
    """Raised when a user already exists in database."""

    def __init__(self, username: str) -> None:
        super().__init__(f"User {username} already exists.")


class InvalidPasswordError(AppError):
    """Raised when an existing user enters an invalid password."""

    def __init__(self) -> None:
        super().__init__("Invalid password.")


class PasswordMismatchError(AppError):
    """Raised when a user enters mismatched passwords."""

    def __init__(self) -> None:
        super().__init__("Passwords do not match.")


class InvalidCredentialsError(AppError):
    """Raised when user provides invalid credentials."""

    def __init__(self) -> None:
        super().__init__("Invalid authentication credentials.")


class UnauthorizedError(AppError):
    """Raised when a user is not authorized to perform an action."""

    def __init__(self) -> None:
        super().__init__("Unauthorized access.")


class ImageDecodingError(Exception):
    """Raised when an image cannot be decoded."""

    def __init__(self) -> None:
        super().__init__("Cannot decode image.")


# 3. Custom HTTP Exceptions for Web Layer
class HTTPError(HTTPException):
    """Base class for HTTP-specific errors."""

    def __init__(self, status_code: int, detail: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class ForbiddenHTTPError(HTTPError):
    """HTTPException raised when a user does not have permission."""

    def __init__(self) -> None:
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not allowed to perform this action.",
        )
