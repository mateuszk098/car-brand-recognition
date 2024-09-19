from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class FileData:
    """Represents a configuration file."""

    NAME: str
    ROOT: str = "network.config"


class Files:
    """Represents available configuration files."""

    ARCH: FileData = FileData(NAME="arch.yaml")
    LOGGING: FileData = FileData(NAME="logging.yaml")
    TRAIN: FileData = FileData(NAME="train.yaml")
