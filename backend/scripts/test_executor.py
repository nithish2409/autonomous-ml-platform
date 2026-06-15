import asyncio
import os
import sys

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import AsyncSessionLocal
from app.services.automation_executor import AutomationExecutor
from app.models.model_registry import ModelRegistry
from sqlalchemy.future import select

async def main():
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(ModelRegistry).where(ModelRegistry.status == "active").limit(1))
        model = res.scalars().first()
        if not model:
            print("No active models")
            return
            
        print(f"Triggering retrain for {model.id}...")
        
        executor = AutomationExecutor()
        
        decision = {
            "action": "retrain",
            "reason": "Test driven retrain",
            "monitoring_metrics": {"drift_score": 0.45, "dataset_size": 2000},
            "training_history": []
        }
        
        result = await executor.execute_decision(str(model.id), decision, db)
        print("RESULT:")
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
