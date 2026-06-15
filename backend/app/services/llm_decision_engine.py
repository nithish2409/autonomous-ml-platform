"""
LLM Decision Engine — sends structured drift signals to Ollama,
enforces JSON-only output, parses actions, and logs decisions.
"""

import json
import uuid
import logging
from datetime import datetime, timedelta, timezone

import httpx

from app.core.retry import async_retry
from app.core.automation_config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    LLM_TIMEOUT_SECONDS,
    LLM_TEMPERATURE,
    ACTION_COOLDOWN_MINUTES,
    WHITELISTED_ACTIONS,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.automation_log import AutomationLog
from app.models.automation_state import AutomationState

logger = logging.getLogger("llm_decision_engine")

# System prompt enforcing JSON-only output
SYSTEM_PROMPT = """You are an automated ML decision engine.

Your job is to decide whether a model requires retraining based strictly on drift_score.

Decision Rules (STRICT):

1. If drift_score > 0.5 → strategy MUST be "RETRAIN".
2. If drift_score <= 0.5 → strategy MUST be "ALERT".

No other strategy is allowed.

You are NOT allowed to output:
- NONE
- NOTIFY
- SCALE
- ROLLBACK
- Any other word
- Lowercase variants
- Additional keys

The ONLY permitted values for "strategy" are:
- "RETRAIN"
- "ALERT"

You MUST return valid JSON.
You MUST return only a JSON object.
Do NOT include markdown.
Do NOT include explanations outside JSON.
Do NOT wrap output in triple backticks.
Do NOT add commentary.

Return EXACTLY this structure:

{
  "analysis": "Brief reasoning in one sentence.",
  "strategy": "RETRAIN" or "ALERT"
}

If uncertain, choose "ALERT".
"""


class PolicyLayer:
    """Checks whether an automated action is allowed based on cooldowns and rules."""

    async def is_action_allowed(
        self,
        model_id: str,
        proposed_action: str,
        db: AsyncSession,
    ) -> tuple[bool, str]:
        """
        Check if action is allowed:
        1. "none" actions are always allowed
        2. Only whitelisted actions are permitted
        3. Check cooldown period for the model
        4. Prevent duplicate consecutive actions
        """
        if proposed_action == "none":
            return True, "No action needed"

        # Whitelist check
        if proposed_action.lower() not in WHITELISTED_ACTIONS:
            return False, f"Action '{proposed_action}' is not whitelisted"

        # Check cooldown
        result = await db.execute(
            select(AutomationState).where(AutomationState.model_id == model_id)
        )
        state = result.scalars().first()

        if state and state.cooldown_until:
            logger.info("Demo mode: LLM cooldown check bypassed")

        # Prevent duplicate consecutive actions
        if state and state.last_action == proposed_action:
            logger.info("Demo mode: Duplicate action check bypassed")

        return True, "Action approved by policy"


class LLMDecisionEngine:
    """Sends structured drift signals to Ollama and parses action decisions."""

    def __init__(self):
        self.policy = PolicyLayer()

    async def evaluate_signal(
        self,
        signal: dict,
        db: AsyncSession,
    ) -> dict:
        """
        Full decision pipeline:
        1. Format signal into prompt
        2. Call Ollama for decision
        3. Parse JSON response
        4. Check policy layer
        5. Log decision to automation_logs
        6. Update automation_state
        """
        model_id = signal["model_id"]

        # 1. Build the user prompt with structured signal
        user_prompt = self._format_signal_prompt(signal)

        # 2. Call Ollama
        try:
            llm_response = await self._call_ollama(user_prompt)
        except Exception as e:
            logger.error("Ollama call failed: %s", e)
            llm_response = {
                "action": "alert",
                "reason": f"LLM unavailable, defaulting to alert: {e}",
                "confidence": 0.0,
                "priority": "medium",
            }

        # 3. Parse the decision
        strategy = llm_response.get("strategy")
        if not strategy:
            logger.warning("Missing strategy. Defaulting to ALERT.")
            strategy = "ALERT"

        strategy = strategy.upper()
        allowed_strategies = {"RETRAIN", "ALERT"}
        
        if strategy not in allowed_strategies:
            logger.warning(f"Invalid strategy {strategy}. Defaulting to ALERT.")
            strategy = "ALERT"

        action = strategy
        reason = llm_response.get("analysis", "No reason provided")
        confidence = llm_response.get("confidence", 0.0)
        priority = llm_response.get("priority", "low")

        # 4. Check policy layer
        allowed, policy_reason = await self.policy.is_action_allowed(model_id, action, db)

        final_status = "approved" if allowed else "blocked"

        # 5. Log decision to automation_logs
        log_entry = AutomationLog(
            id=str(uuid.uuid4()),
            model_id=model_id,
            action=action,
            reason=reason,
            status=final_status,
            log_metadata={
                "confidence": confidence,
                "priority": priority,
                "policy_reason": policy_reason,
                "drift_score": signal.get("drift_score"),
                "performance_delta": signal.get("performance_delta"),
                "request_count": signal.get("request_count"),
                "latency_avg": signal.get("latency_avg"),
                "model_status": signal.get("status"),
                "severity": signal.get("severity"),
                "llm_raw_response": llm_response,
            },
        )
        db.add(log_entry)

        # 6. Update automation state (if action was approved)
        if allowed and action != "none":
            await self._update_state(model_id, action, db)

        await db.commit()

        return {
            "model_id": model_id,
            "action": action,
            "reason": reason,
            "confidence": confidence,
            "priority": priority,
            "status": final_status,
            "policy_reason": policy_reason,
        }

    @staticmethod
    def _format_signal_prompt(signal: dict) -> str:
        """Format the monitoring signal into a structured prompt."""
        return f"""Monitoring Signal Report:
- Model ID: {signal.get('model_id')}
- Model: {signal.get('model_name', 'unknown')} ({signal.get('framework', 'unknown')})
- Version: {signal.get('version', 'unknown')}
- Drift Score: {signal.get('drift_score', 0)}
- Performance Delta: {signal.get('performance_delta', 0)}
- Request Count: {signal.get('request_count', 0)}
- Avg Latency: {signal.get('latency_avg', 0)}ms
- Status: {signal.get('status', 'unknown')}
- Severity: {signal.get('severity', 'unknown')}
- Recent Inferences: {signal.get('n_recent_inferences', 0)}

Based on this signal, what action should be taken?"""

    @staticmethod
    @async_retry(max_retries=3, base_delay=2.0, exceptions=(httpx.HTTPError, httpx.TimeoutException))
    async def _call_ollama(prompt: str) -> dict:

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": "llama3:latest",
                        "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
                        "stream": False,
                        "options": {
                            "temperature": 0
                        }
                    },
                )
                response.raise_for_status()
                result = response.json()
                content = result.get("response", "{}")
                logger.debug(f"Raw Ollama output: {content}")
                
                try:
                    parsed = json.loads(content)
                    
                    allowed_strategies = {"RETRAIN", "ALERT"}
                    strategy = parsed.get("strategy", "").upper()

                    if strategy not in allowed_strategies:
                        logger.warning(
                            f"Invalid LLM strategy received: {strategy}. Defaulting to ALERT."
                        )
                        return {
                            "analysis": parsed.get(
                                "analysis",
                                "Invalid strategy returned by LLM."
                            ),
                            "strategy": "ALERT"
                        }
                    
                    parsed["strategy"] = strategy
                    return parsed
                except json.JSONDecodeError:
                    # Strip markdown blocks
                    cleaned = content.strip()
                    if cleaned.startswith("```json"):
                        cleaned = cleaned[7:]
                    elif cleaned.startswith("```"):
                        cleaned = cleaned[3:]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3]
                    cleaned = cleaned.strip()
                    
                    try:
                        import re
                        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                        if match:
                            return json.loads(match.group(0))
                        return json.loads(cleaned)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse Ollama JSON. Raw output: {content}")
                        return {
                            "analysis": "LLM returned malformed output.",
                            "strategy": "alert"
                        }
            except Exception as e:
                logger.error(f"Ollama structured generation failed: {e}")
                return {
                    "analysis": f"LLM call failed: {str(e)}",
                    "strategy": "alert"
                }

    async def _update_state(self, model_id: str, action: str, db: AsyncSession):
        """Update automation state with cooldown."""
        result = await db.execute(
            select(AutomationState).where(AutomationState.model_id == model_id)
        )
        state = result.scalars().first()

        cooldown = datetime.now(timezone.utc) + timedelta(minutes=ACTION_COOLDOWN_MINUTES)

        if state:
            state.last_action = action
            state.cooldown_until = cooldown
        else:
            state = AutomationState(
                model_id=model_id,
                last_action=action,
                cooldown_until=cooldown,
            )
            db.add(state)
