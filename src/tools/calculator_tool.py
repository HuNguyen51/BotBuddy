"""
Calculator Tool — Tính toán biểu thức toán học.

Tool đơn giản nhưng hữu ích để test LLM tool selection.
Dùng eval() với whitelist an toàn.
"""

from __future__ import annotations

import math

from langchain_core.tools import tool


_SAFE_GLOBALS = {
    "__builtins__": {},
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "int": int,
    "float": float,
    # math functions
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.
    Use this for any calculations. Examples: '2 + 2', 'sqrt(16)', '3.14 * 5**2'."""
    try:
        result = eval(expression, _SAFE_GLOBALS)  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"
