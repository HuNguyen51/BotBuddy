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

from langchain_core.language_models.chat_models import BaseChatModel

from src.utils.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class LLMChatModel:
    """
    Unified LLM interface qua LiteLLM.
    """
    
    @classmethod
    def get(cls, provider: str = "openai", **kwargs) -> BaseChatModel:
        if provider == "openai":
            return cls.__get_openai_chat_model(**kwargs)
        elif provider == "google":
            return cls.__get_google_chat_model(**kwargs)
        elif provider == "litellm":
            return cls.__get_router_chat_model(**kwargs)
        else:
            raise ValueError(f"Invalid provider: {provider}")


    @staticmethod
    def __get_openai_chat_model(
        model = "gpt-oss:120b-cloud", 
        base_url = "https://ollama.com/v1", 
        temperature = 0.0, 
        **kwargs
    ):
        """
        Trả về LangChain-compatible ChatOpenAI instance.
        """
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, base_url=base_url, temperature=temperature, **kwargs)
    
    @staticmethod
    def __get_google_chat_model(
        model = "gemini-2.5-flash", 
        temperature = 0.0, 
        **kwargs
    ):
        """
        Trả về LangChain-compatible ChatGoogleGenerativeAI instance.

        Sử dụng Gemini 2.5 Flash — model nhanh, hỗ trợ tool calling tốt.
        API key đọc tự động từ env var GOOGLE_API_KEY.
        """
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, **kwargs)

    @staticmethod
    def __get_router_chat_model(
        model_list = [
            {
                "model_name": "gpt-oss:120b-cloud",
                "litellm_params": {
                    "model": "ollama/gpt-oss:120b-cloud",
                    "api_base": "http://localhost:11434",
                    "order": 2,
                },
            }], 
        model_fallback = ["gpt-oss:120b-cloud"], 
        temperature = 0.0, 
        **kwargs
    ):
        """
        Trả về LangChain-compatible ChatLiteLLM instance.
        Dùng khi cần pass model vào LangGraph / create_agent.

        model_list = [
            {
                "model_name": "gpt-oss:120b-cloud",
                "litellm_params": {
                    "model": "ollama/gpt-oss:120b-cloud",
                    "api_base": "http://localhost:11434",
                    "order": 2,
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

        model_fallback = ["gpt-oss:120b-cloud", "gemini-2.5-flash"]
        default_model = "gpt-oss:120b-cloud"

        """
        from langchain_litellm import ChatLiteLLMRouter
        from litellm import Router

        router = Router(
            model_list=model_list,
            default_fallbacks=model_fallback,
            enable_pre_call_checks=True, # Required for 'order' to work
            debug_level="WARNING",
        )

        default_model = model_list[0]["model_name"]

        return ChatLiteLLMRouter(
            router = router,
            model = default_model,
            temperature = temperature,
            **kwargs
        )