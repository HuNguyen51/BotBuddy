"""
Research Agent — Agent chuyên biệt cho nghiên cứu.

Kế thừa BaseAgent, thêm:
- System prompt chuyên biệt cho research tasks
- Tools mặc định: web_search + calculator
"""

from __future__ import annotations

from agents.base_agent import BaseAgent
from services.llm_service import LLMService
from tools.calculator_tool import calculator
from tools.web_search_tool import web_search

RESEARCH_SYSTEM_PROMPT = """\
You are a Research Agent — an expert at finding and synthesizing information.

Your capabilities:
1. **Web Search**: Use the web_search tool to find current information
2. **Calculator**: Use the calculator tool for mathematical computations

Instructions:
- Always cite your sources when presenting search results
- Break complex questions into smaller, searchable queries
- Cross-reference information from multiple sources when possible
- If you cannot find reliable information, say so clearly
- Respond in the same language as the user's question
"""


class ResearchAgent(BaseAgent):
    """
    Agent chuyên research — tìm kiếm, tính toán, tổng hợp thông tin.

    Usage::

        agent = ResearchAgent()
        result = agent.invoke("What are the latest AI trends?")
    """

    def __init__(
        self,
        model: Any,
        extra_tools: list | None = None,
        checkpointer: Any = None,
    ) -> None:
    
        tools = [web_search, calculator]
        if extra_tools:
            tools.extend(extra_tools)

        super().__init__(
            model=model,
            tools=tools,
            system_prompt=RESEARCH_SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )
