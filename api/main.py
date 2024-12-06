'''
    This module is the startpoint for running tha API.
'''
from fastapi import FastAPI
from database import engine
from router import posts, upload, users
import models
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


app = FastAPI()

# To create all the tables in the database 
models.Base.metadata.create_all(bind=engine)

# Here we link the routers
app.include_router(posts.router)
app.include_router(users.router)
app.include_router(upload.router)

# Here is code that allows bigger file uploads (100MB)
class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_request_size: int):
        super().__init__(app)
        self.max_request_size = max_request_size

    async def dispatch(self, request: Request, call_next):
        body = await request.body()
        if len(body) > self.max_request_size:
            return Response("Request body too large", status_code=413)
        return await call_next(request)
# Set the max request size to 100 MB
app.add_middleware(RequestSizeLimitMiddleware, max_request_size=100 * 1024 * 1024)