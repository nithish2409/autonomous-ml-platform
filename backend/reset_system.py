
import asyncio
import logging
import os
from sqlalchemy.future import select
from sqlalchemy import text
from app.core.database import AsyncSessionLocal
from app.core.storage import MinioClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reset_system")

async def reset_database():
    logger.info("Resetting Database...")
    async with AsyncSessionLocal() as db:
        # Disable FK checks to allow truncation in any order (safe for full reset)
        # Check if postgres
        try:
            await db.execute(text("TRUNCATE TABLE model_versions, monitoring_metrics, training_jobs, automation_logs, model_registry, datasets CASCADE;"))
            await db.commit()
            logger.info("Database active tables truncated.")
        except Exception as e:
            logger.error(f"Error truncating tables: {e}")
            await db.rollback()

def reset_minio():
    logger.info("Resetting MinIO Storage...")
    try:
        storage = MinioClient()
        # List all objects
        objects = storage.client.list_objects(storage.bucket, recursive=True)
        count = 0
        for obj in objects:
            storage.client.remove_object(storage.bucket, obj.object_name)
            count += 1
        logger.info(f"Deleted {count} objects from MinIO bucket '{storage.bucket}'.")
    except Exception as e:
        logger.error(f"Error resetting MinIO: {e}")

async def main():
    print("WARNING: This will delete ALL data (Models, Datasets, Metrics).")
    await reset_database()
    reset_minio()
    print("System Reset Complete.")

if __name__ == "__main__":
    asyncio.run(main())
