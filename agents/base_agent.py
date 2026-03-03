"""
Base Agent — LangGraph ReAct Agent.

Triển khai ReAct loop (Reasoning + Acting) bằng LangGraph:
  Input → Reason → Choose Tool → Execute → Observe → Loop/Answer

Sử dụng langchain.agents.create_agent để tạo graph,
kết hợp với MemorySaver checkpointer cho conversation state.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from configs.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseAgent:
    """
    ReAct Agent dựa trên LangGraph.

    Usage::

        agent = BaseAgent(
            tools=[web_search, calculator],
            system_prompt="You are a helpful assistant.",
        )
        result = agent.invoke("What is 2+2?")
        result = await agent.ainvoke("Search for AI news")
    """

    def __init__(
        self,
        model: Any,
        tools: list[Any] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        checkpointer: Any | None = None,
    ) -> None:
        self._model = model
        self._tools = tools or []
        self._system_prompt = system_prompt

        # Memory checkpointer — lưu state giữa các turns
        self._checkpointer = checkpointer or MemorySaver()

        # Tạo LangGraph ReAct agent
        self._graph = create_agent(
            model=self._model,
            tools=self._tools,
            system_prompt=self._system_prompt,
            checkpointer=self._checkpointer,
        )

        tool_names = [t.name for t in self._tools] if self._tools else []
        logger.info("Agent initialized — tools=%s, model=%s", tool_names, model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def invoke(
        self,
        message: str,
        *,
        thread_id: str = "default",
    ) -> str:
        """
        Synchronous invoke — gửi message và nhận response.

        Args:
            message: User message
            thread_id: Session ID cho memory (cùng thread_id = cùng conversation)

        Returns:
            Agent response text
        """
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": settings.agent.recursion_limit,
        }

        logger.info("Agent invoke — message='%s', thread_id=%s", message[:100], thread_id)

        result = self._graph.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )

        response = self._extract_response(result)
        logger.info("Agent response — %d chars", len(response))
        return response

    async def ainvoke(
        self,
        message: str,
        *,
        thread_id: str = "default",
    ) -> str:
        """Async invoke."""
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": settings.agent.recursion_limit,
        }

        logger.info("Agent ainvoke — message='%s', thread_id=%s", message[:100], thread_id)

        result = await self._graph.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )

        response = self._extract_response(result)
        logger.info("Agent response — %d chars", len(response))
        return response

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_response(result: dict[str, Any]) -> str:
        """Lấy text response từ LangGraph result."""
        messages = result.get("messages", [])
        if not messages:
            return "No response generated."

        # Lấy message cuối cùng (AI response)
        last_message = messages[-1]
        return last_message.content if hasattr(last_message, "content") else str(last_message)
