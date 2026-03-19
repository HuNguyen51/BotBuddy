"""
Retry — tenacity wrappers cho external API calls.

LLM API timeout, rate limit, network errors là chuyện bình thường.
tenacity xử lý tự động với exponential backoff.
"""

from __future__ import annotations

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
import logging

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# tenacity's before_sleep_log cần stdlib logger
_stdlib_logger = logging.getLogger("tenacity.retry")


def llm_retry(max_attempts: int = 3):
    """
    Retry decorator cho LLM API calls.

    Sẽ retry khi gặp:
    - ConnectionError, TimeoutError
    - API rate limit errors (thường là subclass của Exception)

    Usage::

        @llm_retry(max_attempts=3)
        async def call_llm():
            ...
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=before_sleep_log(_stdlib_logger, logging.WARNING),
        reraise=True,
    )


def api_retry(max_attempts: int = 3):
    """
    Retry decorator cho external API calls (search, etc.)

    Usage::

        @api_retry(max_attempts=2)
        def call_external_api():
            ...
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=before_sleep_log(_stdlib_logger, logging.WARNING),
        reraise=True,
    )
