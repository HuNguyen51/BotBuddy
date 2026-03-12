"""
Chat Endpoint — Streaming & Non-streaming chat API.

Endpoints:
    POST /chat          → Non-streaming response (JSON)
    POST /chat/stream   → Streaming response (SSE)

SSE Event format:
    data: {"type": "token", "content": "Hello", "node": "model"}
    data: {"type": "node_update", "node": "tools"}
    data: {"type": "done", "message": "Stream completed."}
    data: {"type": "error", "detail": "..."}
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from endpoints.schemas import (
    ChatRequest,
    ChatResponse,
    StreamDone,
    StreamError,
    StreamNodeUpdate,
    StreamToken,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


def _get_agent():
    """Lazy-load agent singleton — tránh circular imports & startup delay."""
    from endpoints._agent_singleton import get_agent

    return get_agent()


# ------------------------------------------------------------------
# POST /chat — Non-streaming
# ------------------------------------------------------------------

@router.post(
    "",
    response_model=ChatResponse,
    summary="Chat với Agent (non-streaming)",
    description="Gửi message, nhận full response khi agent xử lý xong.",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """Non-streaming chat — trả về full response."""
    agent = _get_agent()

    logger.info(
        "Chat request — message='%s', thread_id=%s",
        request.message[:100], request.thread_id,
    )

    try:
        response = await agent.ainvoke(
            request.message,
            thread_id=request.thread_id,
        )
    except Exception as e:
        logger.error("Chat error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    logger.info("Chat response — %d chars", len(response))
    return ChatResponse(message=response, thread_id=request.thread_id)


# ------------------------------------------------------------------
# POST /chat/stream — SSE Streaming
# ------------------------------------------------------------------

@router.post(
    "/stream",
    summary="Chat với Agent (SSE streaming)",
    description=(
        "Gửi message, nhận response dưới dạng Server-Sent Events.\n\n"
        "Event types:\n"
        "- `token`: LLM text token (hiển thị real-time)\n"
        "- `node_update`: agent chuyển sang node mới (tool call, reasoning)\n"
        "- `done`: stream hoàn tất\n"
        "- `error`: lỗi xảy ra"
    ),
)
async def chat_stream(request: ChatRequest) -> EventSourceResponse:
    """SSE streaming chat — trả về từng token real-time."""
    agent = _get_agent()

    logger.info(
        "Stream request — message='%s', thread_id=%s",
        request.message[:100], request.thread_id,
    )

    async def event_generator():
        """Generate SSE events từ agent astream."""
        try:
            configurable = {
                "thread_id": request.thread_id,
                "tenant_id": request.tenant_id,
            }
            async for event in agent.astream(
                request.message,
                configurable=configurable,
            ):
                if event["mode"] == "messages":
                    sse_event = StreamToken(
                        content=event["content"],
                        node=event.get("node", ""),
                    )
                    yield {"data": sse_event.model_dump_json()}

                elif event["mode"] == "updates":
                    node_name = event.get("node", "unknown")
                    if node_name == "model":
                        sse_event = StreamNodeUpdate(
                            node=node_name, 
                            state=event["state"]["model"]
                        )
                    elif node_name == "tools":
                        sse_event = StreamNodeUpdate(
                            node=node_name, 
                            state=event["state"]["tools"]
                        )
                    else:
                        sse_event = StreamNodeUpdate(node=node_name, state=event)


                    yield {"data": sse_event.model_dump_json()}

            # Stream hoàn tất
            done_event = StreamDone()
            yield {"data": done_event.model_dump_json()}

        except Exception as e:
            logger.error("Stream error: %s", e)
            error_event = StreamError(detail=str(e), content="Lỗi xảy ra trong quá trình xử lý yêu cầu")
            yield {"data": error_event.model_dump_json()}

    return EventSourceResponse(event_generator())
