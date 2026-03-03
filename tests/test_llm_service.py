"""
Integration Test — Test agent end-to-end.

Cần OPENAI_API_KEY (hoặc API key khác) trong .env để chạy.
Skip tự động nếu không có key.
"""

import os
import pytest


class TestLLMService:
    def test_model_init(self):
        """khởi tạo thành công."""
        from services.llm_service import LLMService

        llm_service = LLMService(model = "ollama/gpt-oss:120b-cloud")
        chat_model = llm_service.get_chat_model()

        respone = chat_model.invoke("bạn là ai")
        assert respone is not None
