import boto3
import os
from datetime import datetime
from logger import get_logger

logger = get_logger("S3Uploader")

class S3Uploader:
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.s3 = boto3.client("s3")

    def upload_file(self, file_path: str, confidence: float):
        try:
            filename = os.path.basename(file_path)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            s3_key = f"low_confidence/{timestamp}_{confidence:.2f}_{filename}"

            self.s3.upload_file(file_path, self.bucket_name, s3_key)

            logger.info(f"Uploaded to S3: s3://{self.bucket_name}/{s3_key}")

        except Exception as e:
            logger.error(f"S3 upload failed: {str(e)}")
