"""
Verify rollback: model now has v1 + v2.
Rollback should switch v2 → v1.
"""
import asyncio

async def main():
    from app.core.database import AsyncSessionLocal
    from app.services.automation_executor import AutomationExecutor
    from app.models.automation_state import AutomationState
    from app.models.model_registry import ModelRegistry
    from sqlalchemy.future import select

    MODEL_ID = "8f66d099-ce42-4d08-9362-ab66c53da088"
    executor = AutomationExecutor()

    async with AsyncSessionLocal() as db:
        # Clear cooldown
        r = await db.execute(select(AutomationState).where(AutomationState.model_id == MODEL_ID))
        s = r.scalars().first()
        if s:
            await db.delete(s)
            await db.commit()

        # Check current version
        r = await db.execute(select(ModelRegistry).where(ModelRegistry.id == MODEL_ID))
        model = r.scalars().first()
        print(f"Before rollback: current_version = {model.current_version}")

        # Execute rollback
        decision = {"action": "rollback", "reason": "Test rollback", "confidence": 0.9, "status": "approved"}
        result = await executor.execute_decision(MODEL_ID, decision, db)
        print(f"Rollback result: {result['status']}")
        if result.get("result"):
            print(f"  {result['result']['old_version']} → {result['result']['new_version']}")

        # Verify DB
        await db.refresh(model)
        print(f"After rollback: current_version = {model.current_version}")

        assert result["status"] == "rolled_back", f"Expected rolled_back, got {result['status']}"
        assert model.current_version == "v1", f"Expected v1, got {model.current_version}"
        print("\n✓ ROLLBACK VERIFIED")

if __name__ == "__main__":
    asyncio.run(main())
