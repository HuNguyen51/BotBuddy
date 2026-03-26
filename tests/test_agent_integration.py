"""
Integration Test — Test agent end-to-end.

Cần OPENAI_API_KEY (hoặc API key khác) trong .env để chạy.
Skip tự động nếu không có key.
"""

import os

import pytest

# Skip tất cả tests nếu không có API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping integration tests",
)

from core.llm import LLMService
llm_service = LLMService(model = "ollama/gpt-oss:120b-cloud")

class TestAgentIntegration:
    """Integration test: agent nhận task → xử lý → trả kết quả."""

    def test_agent_init(self):
        """Agent khởi tạo thành công."""
        from src.agents.base_agent import BaseAgent
        from src.tools.calculator_tool import calculator

        agent = BaseAgent(tools=[calculator],llm_service=llm_service)
        assert agent is not None

    def test_agent_calculator_task(self):
        """Agent gọi calculator tool và trả về kết quả đúng."""
        from src.agents.base_agent import BaseAgent
        from src.tools.calculator_tool import calculator

        agent = BaseAgent(
            tools=[calculator],
            system_prompt="You are a calculator assistant. Use the calculator tool for math.",
            llm_service=llm_service
        )

        result = agent.invoke("What is 15 * 7?")
        assert "105" in result

    def test_research_agent_init(self):
        """ResearchAgent khởi tạo với đúng tools."""
        from src.agents.research_agent import ResearchAgent

        agent = ResearchAgent(llm_service=llm_service)
        assert agent is not None
