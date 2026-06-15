import asyncio
from unittest.mock import MagicMock
import sys
import logging
logger = logging.getLogger("test")

class MockExecutor:
    async def execute_decision(self, model_id: str, decision: dict, db: str):
        print(f"Executing: model_id={model_id}, decision={decision}")

async def main():
    executor = MockExecutor()
    try:
        # Simulate what automation.py does
        await executor.execute_decision("decision-1234", "db_mock")
        print("Success!")
    except Exception as e:
        print(f"Exception caught: {type(e).__name__} - {e}")

if __name__ == "__main__":
    asyncio.run(main())
