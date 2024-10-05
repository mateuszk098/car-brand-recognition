import os

import uvicorn
from fastapi import FastAPI

from brands.web.user import router as user_router

PORT = int(os.getenv("FASTAPI_PORT", 8000))
HOST = os.getenv("FASTAPI_HOST", "localhost")
RELOAD = False if os.getenv("MODE") == "PROD" else True

app = FastAPI()
app.include_router(user_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=RELOAD)
