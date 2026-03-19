"""
Prompts Package — Tập trung quản lý tất cả system prompts.

Tất cả prompts được định nghĩa ở đây, các nơi khác chỉ import và dùng.
Giúp dễ xem xét, chỉnh sửa, và quản lý prompt mà không cần sửa code logic.

Usage::

    from src.prompts import FNB_SYSTEM_PROMPT, RESEARCH_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT

    agent = BaseAgent(system_prompt=FNB_SYSTEM_PROMPT)
"""

from src.prompts.base_prompt import DEFAULT_SYSTEM_PROMPT
from src.prompts.fnb_prompt import FNB_SYSTEM_PROMPT
from src.prompts.research_prompt import RESEARCH_SYSTEM_PROMPT

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "FNB_SYSTEM_PROMPT",
    "RESEARCH_SYSTEM_PROMPT",
]
