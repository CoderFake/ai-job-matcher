from fastapi import APIRouter
from .endpoints import cv, jobs, users

# Tạo router API chính
api_router = APIRouter()

# Đăng ký các router con
api_router.include_router(cv.router, prefix="/cv", tags=["CV"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["Jobs"])
api_router.include_router(users.router, prefix="/users", tags=["Users"])