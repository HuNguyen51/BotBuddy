"""
Settings Module — Pydantic BaseSettings cho toàn bộ hệ thống.

Load config từ 2 nguồn (ưu tiên từ trên xuống):
1. Environment variables / .env file
2. configs/model_config.yaml

Tất cả config đều type-safe và validated tự động bởi Pydantic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

# Đường dẫn gốc của project
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"


def _load_yaml_config() -> dict[str, Any]:
    """Load model_config.yaml, trả về dict rỗng nếu file không tồn tại."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


_yaml = _load_yaml_config()


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class LLMSettings(BaseSettings):
    """Config cho LLM provider (qua LiteLLM)."""
    model: str = Field(default=_yaml.get("llm", {}).get("model", "must put llm `model` in `model_config.yaml` file"))
    temperature: float = Field(default=_yaml.get("llm", {}).get("temperature", 0.0))
    max_tokens: int = Field(default=_yaml.get("llm", {}).get("max_tokens", 128000))


class AgentSettings(BaseSettings):
    """Config cho Agent orchestration."""
    max_iterations: int = Field(
        default=_yaml.get("agent", {}).get("max_iterations", 10)
    )
    recursion_limit: int = Field(
        default=_yaml.get("agent", {}).get("recursion_limit", 25)
    )


class VectorStoreSettings(BaseSettings):
    """Config cho ChromaDB vector store."""
    collection_name: str = Field(
        default=_yaml.get("vector_store", {}).get("collection_name", "agent_memory")
    )
    persist_directory: str = Field(
        default=_yaml.get("vector_store", {}).get("persist_directory", "./data/chroma_db")
    )


class LoggingSettings(BaseSettings):
    """Config cho structured logging."""
    level: str = Field(default=_yaml.get("logging", {}).get("level", "INFO"))
    json_output: bool = Field(
        default=_yaml.get("logging", {}).get("json_output", True)
    )


# ---------------------------------------------------------------------------
# Root Settings — single entry point
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Root config cho toàn bộ hệ thống.

    Usage:
        from configs.settings import settings
        print(settings.llm.model)       # "gpt-4o"
        print(settings.openai_api_key)  # từ .env
    """

    # API Keys (từ environment / .env)
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", alias="LANGFUSE_HOST"
    )

    # Sub-configs
    llm: LLMSettings = Field(default_factory=LLMSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


# Singleton instance — import settings từ đây
settings = Settings()
