
import asyncio
import os
from dotenv import load_dotenv

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select

from app.models.dataset import Dataset
from app.models.model_registry import ModelRegistry

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

import json

async def main():
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Dataset))
        datasets = result.scalars().all()
        
        result = await db.execute(select(ModelRegistry))
        models = result.scalars().all()
        
        data = {
            "datasets": [{"id": str(d.id), "name": d.name} for d in datasets],
            "models": [{
                "id": str(m.id), 
                "dataset_id": str(m.dataset_id), 
                "status": m.status, 
                "version": m.current_version
            } for m in models]
        }
        print(json.dumps(data, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
