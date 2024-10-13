"""Main module of the application."""

import os
from pathlib import Path

from fastapi import FastAPI

from app.service.utils import on_startup
from app.web.routers.admin import router as admin_router
from app.web.routers.user import router as user_router

DEBUG = os.getenv("DEBUG", "False").lower() == "true"

description_file = Path(__file__).resolve().with_name("description.md")
with open(description_file, "r", encoding="utf-8") as f:
    description = f.read()

app = FastAPI(
    debug=DEBUG,
    title="Car Brand Recognition",
    description=description,
    version="0.1.0",
    lifespan=on_startup,
    contact={
        "name": "Mateusz Kowalczyk",
        "url": "https://github.com/mateuszk098",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)
app.include_router(user_router)
app.include_router(admin_router)
