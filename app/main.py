import os

import uvicorn
from fastapi import FastAPI

from app.data import init_db
from app.web.routers.user import router as user_router

PORT = int(os.getenv("FASTAPI_PORT", 8000))
HOST = os.getenv("FASTAPI_HOST", "localhost")

app = FastAPI()
app.include_router(user_router)


init_db()

if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
