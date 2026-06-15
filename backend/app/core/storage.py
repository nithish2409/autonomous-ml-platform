from minio import Minio
from minio.error import S3Error
import os

from app.core.retry import sync_retry


class MinioClient:
    def __init__(self):
        self.client = Minio(
            endpoint=os.getenv("MINIO_ENDPOINT"),
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            secure=False
        )
        self.bucket = os.getenv("MINIO_BUCKET")

    @sync_retry(max_retries=3, base_delay=1.0, exceptions=(S3Error, Exception))
    def ensure_bucket(self):
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    @sync_retry(max_retries=3, base_delay=1.0, exceptions=(S3Error, Exception))
    def upload_file(self, object_name, file_path):
        self.client.fput_object(self.bucket, object_name, file_path)

    @sync_retry(max_retries=3, base_delay=1.0, exceptions=(S3Error, Exception))
    def upload_bytes(self, object_name, data, length):
        self.client.put_object(self.bucket, object_name, data, length)

