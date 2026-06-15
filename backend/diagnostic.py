import asyncio
import logging
from app.core.database import AsyncSessionLocal
from app.services.monitoring_service import MonitoringService
from app.services.llm_decision_engine import LLMDecisionEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diagnostic")

async def run_diagnostics():
    logger.info("Starting diagnostic monitoring cycle...")
    async with AsyncSessionLocal() as db:
        service = MonitoringService()
        engine = LLMDecisionEngine()
        signals = await service.check_all_active_models(db)
        
        logger.info(f"Generated {len(signals)} signals.")
        for s in signals:
            logger.info(f"Signal: {s}")
            if "error" not in s:
                decision = await engine.evaluate_signal(s, db)
                logger.info(f"Decision: {decision}")

if __name__ == "__main__":
    asyncio.run(run_diagnostics())
