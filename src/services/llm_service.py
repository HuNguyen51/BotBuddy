"""
LLM Service — LiteLLM Wrapper.

Đổi provider chỉ bằng cách đổi model string trong config:
  - "gpt-4o"                → OpenAI
  - "claude-3-5-sonnet-..."  → Anthropic
  - "ollama/llama3"          → Local Ollama
  - "gemini/gemini-1.5-pro"  → Google

Không cần sửa bất kỳ logic code nào.
"""

from __future__ import annotations

from typing import Any

import litellm

from src.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class LLMService:
    """
    Unified LLM interface qua LiteLLM.
    """

    def __init__(
        self,
        temperature: float | None = None,
    ) -> None:
        self.temperature = temperature if temperature is not None else settings.llm.temperature

        logger.info("LLM Service initialized — temperature=%s", self.temperature)
    
    def get_chat_model(self, provider: str = "litellm"):
        if provider == "openai":
            return self.get_openai_chat_model()
        elif provider == "google":
            return self.get_google_chat_model()
        elif provider == "litellm":
            return self.get_router_chat_model()
        else:
            raise ValueError(f"Invalid provider: {provider}")

    def get_openai_chat_model(self):
        """
        Trả về LangChain-compatible ChatOpenAI instance.
        """
        from langchain_openai import ChatOpenAI

        DEFAULT_MODEL = "gpt-oss:120b-cloud"
        DEFAULT_BASE_URL = "https://ollama.com/v1"

        return ChatOpenAI(
            model=DEFAULT_MODEL,
            temperature=self.temperature,
            base_url=DEFAULT_BASE_URL,
        )
    
    def get_google_chat_model(self):
        """
        Trả về LangChain-compatible ChatGoogleGenerativeAI instance.

        Sử dụng Gemini 2.5 Flash — model nhanh, hỗ trợ tool calling tốt.
        API key đọc tự động từ env var GOOGLE_API_KEY.
        """
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=self.temperature,
        )

    def get_router_chat_model(self):
        """
        Trả về LangChain-compatible ChatLiteLLM instance.
        Dùng khi cần pass model vào LangGraph / create_agent.
        """
        from langchain_litellm import ChatLiteLLMRouter
        from litellm import Router

        MODEL_LIST = [
            {
                "model_name": "gpt-oss:120b-cloud",
                "litellm_params": {
                    "model": "ollama/gpt-oss:120b-cloud",
                    "api_base": "http://localhost:11434",
                    "order": 2,  # 👈 Highest priority
                },
            },
            {
                "model_name": "gpt-oss:120b-cloud",
                "litellm_params": {
                    "model": "openai/gpt-oss:120b-cloud",
                    "api_base": "https://ollama.com/v1",
                    "order": 1, 
                },
            },
            {
                "model_name": "gemini-2.5-flash",
                "litellm_params": {
                    "model": "gemini/gemini-2.5-flash",
                    "order": 3, 
                },
            },
        ]

        MODEL_FALLBACK = ["gpt-oss:120b-cloud", "gemini-2.5-flash"]
        DEFAULT_MODEL = "gpt-oss:120b-cloud"

        litellm_router = Router(
            model_list=MODEL_LIST,
            default_fallbacks=MODEL_FALLBACK,
            enable_pre_call_checks=True, # Required for 'order' to work
            debug_level="INFO",
            set_verbose=True,
        )

        # ChatLiteLLM for multi-provider
        return ChatLiteLLMRouter(
            router=litellm_router, 
            model=DEFAULT_MODEL,
            temperature=self.temperature,
            mock_testing_fallbacks=True
        )