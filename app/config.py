from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    CORS_ALLOW_ORIGINS: List[str] = [
        "http://localhost:3000",  
        "http://127.0.0.1:3000",
        "https://skazochnik.vercel.app",
    ]

    database_url: str

    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str
    JWT_EXPIRE_HOURS: int

    HF_TOKEN: str

    YANDEX_CLOUD_API_KEY: str
    YANDEX_CLOUD_PROJECT: str

    S3_BUCKET = str

    S3_ACCESS_KEY = str
    S3_SECRET_KEY = str

    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
