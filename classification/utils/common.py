import logging
import logging.config
from importlib.resources import files
from logging import Logger

import yaml


def init_logger(name: str | None = None) -> Logger:
    config_path = files("classification.config").joinpath("logging.yaml")
    with open(str(config_path), "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    logging.config.dictConfig(content)
    return logging.getLogger(name)
