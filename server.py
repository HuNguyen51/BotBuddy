"""
Main Entry Point — FastAPI Server.

Cách chạy:
    uv run uvicorn main:app --reload --port 8000

Endpoints:
    POST /chat          → Non-streaming chat
    POST /chat/stream   → SSE streaming chat
    GET  /health        → Health check
    GET  /docs          → Swagger UI
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from utils.logger import setup_logger

# Setup ROOT logger
logger = setup_logger(None, log_file="logs/app.log")


# ------------------------------------------------------------------
# Lifespan — startup / shutdown events
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: warm-up agent. Shutdown: cleanup."""
    logger.info("🚀 ScanX Agent API starting...")

    # Warm-up: khởi tạo agent trước khi nhận request
    from endpoints._agent_singleton import get_agent
    get_agent()

    logger.info("✅ Agent ready — accepting requests.")
    yield
    logger.info("👋 ScanX Agent API shutting down.")


# ------------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------------

app = FastAPI(
    title="ScanX Agent API",
    description=(
        "AI Agent API với streaming support.\n\n"
        "- **POST /chat** — Full response\n"
        "- **POST /chat/stream** — Real-time SSE streaming\n"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — cho phép frontend từ mọi origin (dev mode)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

from endpoints.chat import router as chat_router  # noqa: E402

app.include_router(chat_router)


@app.get("/health", tags=["System"])
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "scanx-agent"}
