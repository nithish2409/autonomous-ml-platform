import asyncio
import os
import sys

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import AsyncSessionLocal
from app.models.model_version import ModelVersion
from sqlalchemy.future import select

async def main():
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(ModelVersion).order_by(ModelVersion.version_number))
        versions = res.scalars().all()
        for v in versions:
            print(f"version: {v.version_number} | model_id: {v.model_id} | metrics: {v.metrics != None} | created: {v.created_at}")

if __name__ == "__main__":
    asyncio.run(main())
