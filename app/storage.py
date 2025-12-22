import uuid
from botocore.exceptions import ClientError
import boto3
import os



S3_ENDPOINT = "https://storage.yandexcloud.net"
S3_REGION="ru-central1"
S3_BUCKET = os.getenv("S3_BUCKET")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
PUBLIC_CDN_URL="https://storage.yandexcloud.net/storygrad"


class ImageStorage:
    def __init__(self):
        self.client = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            region_name=S3_REGION,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
        )

    def upload_image(self, file_bytes: bytes, content_type: str) -> str:
        """
        Загружает картинку и возвращает публичный URL
        """
        key = f"images/{uuid.uuid4()}.png"

        try:
            self.client.put_object(
                Bucket=S3_BUCKET,
                Key=key,
                Body=file_bytes,
                ContentType=content_type,
            )
        except ClientError as e:
            raise RuntimeError("Image upload failed") from e

        return f"{PUBLIC_CDN_URL}/{key}"


# singleton-экземпляр
image_storage = ImageStorage()