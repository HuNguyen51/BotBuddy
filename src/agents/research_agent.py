"""
Research Agent — Agent chuyên biệt cho nghiên cứu.

Kế thừa BaseAgent, thêm:
- System prompt chuyên biệt cho research tasks
- Tools mặc định: web_search + calculator
"""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseAgent
from src.tools.calculator_tool import calculator
from src.tools.web_search_tool import web_search

from src.prompts.research_prompt import RESEARCH_SYSTEM_PROMPT



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
