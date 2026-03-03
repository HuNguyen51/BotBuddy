"""
Base Tool — Pydantic Schema cho tool definitions.

Mỗi tool cần:
- name + description → LLM dựa vào đây để quyết định gọi tool nào
- args_schema → Pydantic model mô tả input parameters
- _run() → Logic thực thi

Tương thích với LangChain @tool decorator pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Kết quả trả về từ tool execution."""
    success: bool = True
    output: str = ""
    error: str | None = None


class BaseTool(ABC):
    """
    Abstract base cho tất cả tools.

    Subclass cần override:
    - name, description (class attributes)
    - args_schema (Pydantic model cho input)
    - _run() method
    """

    name: str = "base_tool"
    description: str = "Base tool description"

    @abstractmethod
    def _run(self, **kwargs: Any) -> str:
        """Implement tool logic tại đây."""
        ...

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute tool với error handling."""
        try:
            output = self._run(**kwargs)
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
