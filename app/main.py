import os

import uvicorn
from fastapi import FastAPI

from app.service.utils import on_startup
from app.web.routers.admin import router as admin_router
from app.web.routers.user import router as user_router

PORT = int(os.getenv("FASTAPI_PORT", 8000))
HOST = os.getenv("FASTAPI_HOST", "localhost")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

with open("description.md", "r", encoding="utf-8") as f:
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


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
