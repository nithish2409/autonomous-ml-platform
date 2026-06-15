"""
Retry decorators with exponential backoff for external service calls.
"""

import asyncio
import functools
import logging
import time
from typing import TypeVar, Callable

logger = logging.getLogger("retry")

F = TypeVar("F", bound=Callable)


def sync_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """Decorator: retries a synchronous function with exponential backoff."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        delay = base_delay * (backoff_factor ** (attempt - 1))
                        logger.warning(
                            "Retry %d/%d for %s after %.1fs — %s",
                            attempt, max_retries, func.__name__, delay, exc,
                        )
                        time.sleep(delay)
            raise last_exc  # type: ignore[misc]
        return wrapper  # type: ignore[return-value]
    return decorator


def async_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """Decorator: retries an async function with exponential backoff."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        delay = base_delay * (backoff_factor ** (attempt - 1))
                        logger.warning(
                            "Retry %d/%d for %s after %.1fs — %s",
                            attempt, max_retries, func.__name__, delay, exc,
                        )
                        await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc]
        return wrapper  # type: ignore[return-value]
    return decorator
