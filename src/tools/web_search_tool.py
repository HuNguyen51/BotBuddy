"""
Web Search Tool — Tìm kiếm thông tin trên internet.

Sử dụng duckduckgo-search (miễn phí, không cần API key).
Được wrap bằng LangChain @tool decorator để tương thích với LangGraph.
"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for current information. Use this when you need to find
    up-to-date facts, news, documentation, or answers that you don't know."""
    from ddgs import DDGS

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return "No results found."

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"[{i}] {r.get('title', 'No title')}\n"
                f"    URL: {r.get('href', 'N/A')}\n"
                f"    {r.get('body', 'No description')}"
            )
        return "\n\n".join(formatted)

    except Exception as e:
        return f"Search error: {e}"

