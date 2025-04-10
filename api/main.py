'''
    This module is the startpoint for running tha API.
'''
from fastapi import FastAPI
from celery import Celery
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from api.config import settings
import api.models as models
from api.database import engine
from api.router import upload, users, tasks, recordings


app = FastAPI()

celery = Celery(
    __name__,
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)
celery.conf.broker_connection_retry_on_startup = True

# To create all the tables in the database 
models.Base.metadata.create_all(bind=engine)

# Here we link the routers
app.include_router(users.router)
app.include_router(upload.router)
app.include_router(tasks.router)
app.include_router(recordings.router)

# Here is code that allows bigger file uploads (100MB)
class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_request_size: int):
        super().__init__(app)
        self.max_request_size = max_request_size

    async def dispatch(self, request: Request, call_next):
        body = await request.body()
        if len(body) > self.max_request_size:
            print(f"Request body with size {len(body)}")
            return Response("Request body too large", status_code=413)
        return await call_next(request)
# Set the max request size to 200 MB
app.add_middleware(RequestSizeLimitMiddleware, max_request_size=1000 * 1024 * 1024)