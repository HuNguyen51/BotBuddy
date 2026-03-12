"""
Base Agent — LangGraph ReAct Agent.

Triển khai ReAct loop (Reasoning + Acting) bằng LangGraph:
  Input → Reason → Choose Tool → Execute → Observe → Loop/Answer

Sử dụng langchain.agents.create_agent để tạo graph,
kết hợp với MemorySaver checkpointer cho conversation state.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessageChunk, HumanMessage
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
        configurable: dict[str, Any] = {},
    ) -> str:
        """
        Synchronous invoke — gửi message và nhận response.

        Args:
            message: User message
            configurable: Cấu hình cho agent

        Returns:
            Agent response text
        """
        config = {
            "configurable": configurable,
            "recursion_limit": settings.agent.recursion_limit,
        }

        logger.info("Agent invoke — message='%s', configurable=%s", message[:100], configurable)

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
        configurable: dict[str, Any] = {},
    ) -> str:
        """Async invoke."""
        config = {
            "configurable": configurable,
            "recursion_limit": settings.agent.recursion_limit,
        }

        logger.info("Agent ainvoke — message='%s', configurable=%s", message[:100], configurable)

        result = await self._graph.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )

        response = self._extract_response(result)
        logger.info("Agent response — %d chars", len(response))
        return response

    async def astream(
        self,
        message: str,
        *,
        configurable: dict[str, Any] = {},
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Async stream — kết hợp cả messages + updates.

        Yield dict với format:
            {"mode": "messages", "content": "token text"}
            {"mode": "updates", "node": "agent", "state": {...}}

        Phù hợp cho UI cần vừa hiển thị streaming text,
        vừa tracking trạng thái từng node (tool calls, reasoning, etc.)
        """
        config = {
            "configurable": configurable,
            "recursion_limit": settings.agent.recursion_limit,
        }

        logger.info("Agent astream — message='%s', configurable=%s", message[:100], configurable)

        async for mode, chunk in self._graph.astream(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                msg, metadata = chunk
                if (
                    isinstance(msg, AIMessageChunk)
                    and msg.content
                    and not msg.tool_calls
                    and not msg.tool_call_chunks
                ):
                    text = self._normalize_content(msg.content)
                    if text:
                        yield {
                            "mode": "messages",
                            "node": metadata.get("langgraph_node", ""),
                            "content": text,
                        }

            elif mode == "updates":
                node_name = next(iter(chunk), None) if chunk else None
                yield {
                    "mode": "updates",
                    "node": node_name,
                    "state": chunk,
                }

        logger.info("Agent astream complete — configurable=%s", configurable)

    async def astream_messages(
        self,
        message: str,
        *,
        configurable: dict[str, Any] = {},
    ) -> AsyncIterator[str]:
        """
        Stream LLM tokens — từng token text của AI response.

        Yield: str (từng đoạn text nhỏ)
        Phù hợp cho chat UI cần hiển thị typewriter effect.
        """
        config = {
            "configurable": configurable,
            "recursion_limit": settings.agent.recursion_limit,
        }

        logger.info(
            "Agent astream_messages — message='%s', configurable=%s",
            message[:100], configurable,
        )

        async for chunk, metadata in self._graph.astream(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            stream_mode="messages",
        ):
            # Chỉ yield AI text chunks (không phải tool calls hay tool results)
            if (
                isinstance(chunk, AIMessageChunk)
                and chunk.content
                and not chunk.tool_calls
                and not chunk.tool_call_chunks
            ):
                text = self._normalize_content(chunk.content)
                if text:
                    yield text

        logger.info("Agent astream_messages complete — configurable=%s", configurable)

    async def astream_updates(
        self,
        message: str,
        *,
        configurable: dict[str, Any] = {},
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream state updates — sau mỗi node step của graph.

        Yield: dict với format:
            {
                "node": "agent" | "tools" | ...,
                "messages": [...],  # messages mới được thêm ở step này
            }

        Phù hợp cho debugging, monitoring, hoặc UI cần hiển thị
        từng bước reasoning của agent (tool calls → results → final answer).
        """
        config = {
            "configurable": configurable,
            "recursion_limit": settings.agent.recursion_limit,
        }

        logger.info(
            "Agent astream_updates — message='%s', configurable=%s",
            message[:100], thread_id,
        )

        async for update in self._graph.astream(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            stream_mode="updates",
        ):
            # update format: {"node_name": {"messages": [...]}}
            node_name = next(iter(update), None)
            if node_name is None:
                continue

            node_state = update[node_name]
            messages = node_state.get("messages", [])

            yield {
                "node": node_name,
                "messages": messages,
            }

            logger.debug(
                "Node '%s' completed — %d messages",
                node_name, len(messages),
            )

        logger.info("Agent astream_updates complete — thread_id=%s", thread_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_content(content: str | list[Any]) -> str:
        """
        Normalize message content thành plain text.

        Gemini trả content dạng mixed:
            - str: "hello world"                          (hầu hết các chunks)
            - list: [{'type':'text', 'text':'...', 'extras': {...}}, ...]  (chunk đầu)

        OpenAI luôn trả str.
        Method này đảm bảo output luôn là str thuần túy.
        """
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts)

        return str(content)

    def _extract_response(self, result: dict[str, Any]) -> str:
        """Lấy text response từ LangGraph result."""
        messages = result.get("messages", [])
        if not messages:
            return "No response generated."

        # Lấy message cuối cùng (AI response)
        last_message = messages[-1]
        if hasattr(last_message, "content"):
            return self._normalize_content(last_message.content)
        return str(last_message)
