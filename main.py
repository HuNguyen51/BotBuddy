import multiprocessing.popen_spawn_posix
import multiprocessing.popen_spawn_posix
import multiprocessing.popen_spawn_posix
import multiprocessing.popen_spawn_posix
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

def _get_agent():
    """Lazy-load agent singleton — tránh circular imports & startup delay."""
    from endpoints._agent_singleton import get_agent

    return get_agent()

async def chat_stream(request: ChatRequest):
    """SSE streaming chat — trả về từng token real-time."""
    agent = _get_agent()

    logger.info(
        "Stream request — message='%s', thread_id=%s",
        request.message[:100], request.thread_id,
    )

    async def event_generator():
        """Generate SSE events từ agent astream."""
        try:
            async for event in agent.astream(
                request.message,
                thread_id=request.thread_id,
            ):
                if event["mode"] == "messages":
                    node_name = event.get("node", "unknown")

                    sse_event = StreamToken(
                        node=node_name,
                        content=event["content"],
                    )
                    yield sse_event

                elif event["mode"] == "updates":
                    node_name = event.get("node", "unknown")
                    if node_name == "model":
                        sse_event = StreamNodeUpdate(
                            node=node_name, 
                            state=event["state"]['model']
                        )
                    elif node_name == "tools":
                        sse_event = StreamNodeUpdate(
                            node=node_name, 
                            state=event["state"]["tools"]
                        )
                    else:
                        sse_event = StreamNodeUpdate(node=node_name, state=event)

                    yield sse_event

            # Stream hoàn tất
            done_event = StreamDone()
            yield done_event

        except Exception as e:
            logger.error("Stream error: %s", e)
            error_event = StreamError(detail=str(e))
            yield error_event

    return event_generator()

async def run(message: str, thread_id: str):
    request = ChatRequest(message=message, thread_id=thread_id)
    generator = await chat_stream(request)
    async for event in generator:
        event_type = event.type
        if event_type == "token":
            print(event.content, end="", flush=True)
        elif event_type == "node_update":
            if event.node == "model":
                if event.state['messages'][-1].response_metadata['finish_reason'] == "stop":
                    print()
            print("---",event.state, "---")
        elif event_type == "done":
            print("---",event.message, "---")
        elif event_type == "error":
            print("---",event.detail, "---")
        else:
            print(event)
    

if __name__ == "__main__":
    import asyncio
    asyncio.run(run("sử dụng tool để giải các bài toán bên dưới: \n4+11*2=?", "session-1"))
    asyncio.run(run("11-12+3*9=?", "session-1"))
    print("="*50)
    asyncio.run(run("11-12+3*9=?", "session-2"))