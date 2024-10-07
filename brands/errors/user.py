"""Custom exceptions for user errors."""


class MissingUserError(Exception):
    """Raised when a user is not found in database."""

    def __init__(self, message="User not found") -> None:
        self.message = message
        super().__init__(self.message)


class UserAlreadyExistsError(Exception):
    """Raised when a user already exists in database."""

    def __init__(self, message="User already exists") -> None:
        self.message = message
        super().__init__(self.message)


class IncorrectPasswordError(Exception):
    """Raised when an existing user enters an incorrect password."""

    def __init__(self, message="Incorrect password") -> None:
        self.message = message
        super().__init__(self.message)
