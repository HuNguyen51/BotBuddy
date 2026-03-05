"""
Endpoint Schemas — Pydantic models cho request/response.

Định nghĩa tất cả data contracts giữa Client và Server.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Request
# ------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Request body cho chat endpoint."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="User message gửi đến agent.",
        examples=["giá vàng hôm nay", "What is 2+2?"],
    )
    thread_id: str = Field(
        default="default",
        description="Session ID — cùng thread_id = cùng conversation history.",
        examples=["user-123-session-1"],
    )


# ------------------------------------------------------------------
# Response (non-streaming)
# ------------------------------------------------------------------

class ChatResponse(BaseModel):
    """Response body cho non-streaming chat endpoint."""

    message: str = Field(
        ...,
        description="Agent response text.",
    )
    thread_id: str = Field(
        ...,
        description="Session ID đã sử dụng.",
    )


# ------------------------------------------------------------------
# SSE Event types (streaming)
# ------------------------------------------------------------------

class StreamToken(BaseModel):
    """SSE event: một token text từ LLM."""

    type: str = "token"
    node: str = Field(default="", description="Graph node đang chạy.")
    content: str = Field(..., description="Token text content.")


class StreamNodeUpdate(BaseModel):
    """SSE event: một node trong graph đã hoàn thành."""

    type: str = "node_update"
    node: str = Field(..., description="Tên node vừa hoàn thành.")
    state: dict = Field(..., description="Content of the node.")


class StreamError(BaseModel):
    """SSE event: lỗi xảy ra trong quá trình streaming."""

    type: str = "error"
    detail: str = Field(..., description="Mô tả lỗi.")


class StreamDone(BaseModel):
    """SSE event: streaming đã hoàn tất."""

    type: str = "done"
    message: str = Field(
        default="Stream completed.",
        description="Completion message.",
    )
