"""
Agent Singleton — khởi tạo agent một lần duy nhất.

Tránh tạo lại model + graph mỗi request.
Thread-safe cho async FastAPI.
"""

from __future__ import annotations

from functools import lru_cache

from agents.research_agent import ResearchAgent
from memory.conversation_memory import memory
from services.llm_service import LLMService
from utils.logger import setup_logger

logger = setup_logger(__name__)


@lru_cache(maxsize=1)
def get_agent() -> ResearchAgent:
    """
    Tạo và cache ResearchAgent singleton.

    Gọi lần đầu → khởi tạo model + agent.
    Gọi lần sau → trả về instance đã cache.
    """
    logger.info("Initializing Agent singleton...")

    model = LLMService().get_chat_model(provider="openai")
    agent = ResearchAgent(model=model, checkpointer=memory)

    logger.info("Agent singleton ready.")
    return agent
