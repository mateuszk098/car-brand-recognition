class MissingUserError(Exception):
    def __init__(self, message="User not found") -> None:
        self.message = message
        super().__init__(self.message)


class UserAlreadyExistsError(Exception):
    def __init__(self, message="User already exists") -> None:
        self.message = message
        super().__init__(self.message)
