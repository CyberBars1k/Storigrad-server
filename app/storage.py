import uuid
from botocore.exceptions import ClientError
import boto3
import os



S3_ENDPOINT = "https://storage.yandexcloud.net"
S3_REGION = "ru-central1"
S3_BUCKET = os.getenv("S3_BUCKET")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
# You can override this with a CDN / custom domain if you have one.
PUBLIC_CDN_URL = os.getenv("PUBLIC_CDN_URL") or (
    f"{S3_ENDPOINT}/{S3_BUCKET}" if S3_BUCKET else S3_ENDPOINT
)

# Basic upload limits
MAX_IMAGE_SIZE_BYTES = int(os.getenv("MAX_IMAGE_SIZE_BYTES") or str(5 * 1024 * 1024))
ALLOWED_IMAGE_EXTS = {"png", "jpg", "jpeg", "webp"}
DISALLOWED_CONTENT_TYPES = {"image/svg+xml"}


class ImageStorage:
    def __init__(self):
        if not S3_BUCKET:
            raise RuntimeError("S3_BUCKET is not set")
        if not S3_ACCESS_KEY:
            raise RuntimeError("S3_ACCESS_KEY is not set")
        if not S3_SECRET_KEY:
            raise RuntimeError("S3_SECRET_KEY is not set")

        self.client = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            region_name=S3_REGION,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
        )

    def upload_image(self, file_bytes: bytes, content_type: str) -> str:
        """Загружает картинку и возвращает публичный URL."""

        if content_type in DISALLOWED_CONTENT_TYPES:
            raise ValueError("Unsupported image type")

        if len(file_bytes) > MAX_IMAGE_SIZE_BYTES:
            raise ValueError("Image too large")

        # Derive extension from content-type
        # e.g. image/jpeg -> jpeg
        ext = (content_type or "").split("/")[-1].lower().strip()
        if ext == "jpg":
            ext = "jpeg"

        if ext not in ALLOWED_IMAGE_EXTS:
            raise ValueError("Unsupported image type")

        key = f"images/{uuid.uuid4()}.{ext}"

        try:
            self.client.put_object(
                Bucket=S3_BUCKET,
                Key=key,
                Body=file_bytes,
                ContentType=content_type,
                ACL="public-read",
            )
        except ClientError as e:
            raise RuntimeError(e.response["Error"]) from e

        return f"{PUBLIC_CDN_URL}/{key}"


# singleton-экземпляр
image_storage = ImageStorage()