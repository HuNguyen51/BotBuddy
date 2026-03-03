"""
Main Entry Point — Agentic AI System.

Interactive chat với Research Agent qua terminal.

Cách chạy:
    uv run python main.py
    
Gõ tin nhắn → Enter → Agent trả lời.
Gõ 'exit' hoặc Ctrl+C để thoát.
"""

from __future__ import annotations

from utils.logger import setup_logger

# Setup ROOT logger TRƯỚC mọi thứ khác (name=None → root logger)
logger = setup_logger(None, log_file="logs/app.log")

BANNER = """
╔══════════════════════════════════════════════╗
║          🤖 ScanX AI Agent Chat              ║
║                                              ║
║  Gõ tin nhắn rồi nhấn Enter để trò chuyện.  ║
║  Gõ 'exit' hoặc Ctrl+C để thoát.            ║
╚══════════════════════════════════════════════╝
"""


def main() -> None:
    """Interactive chat loop với Research Agent."""
    from services.llm_service import LLMService # brain
    from agents.research_agent import ResearchAgent # body
    from memory.conversation_memory import memory # memory

    logger.info("Agentic AI System starting...")

    model = LLMService().get_chat_model(provider="openai") # brain_init
    agent = ResearchAgent(model=model, checkpointer=memory) # body_init with brain and memory
    
    thread_id = "interactive-session"

    print(BANNER)

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Tạm biệt!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("\n👋 Tạm biệt!")
            break

        logger.info("User: %s", user_input)

        try:
            response = agent.invoke(user_input, thread_id=thread_id)
            logger.info("Agent: %s", response[:200])
            print(f"\n🤖 Agent: {response}")
        except Exception as e:
            logger.error("Agent error: %s", e)
            print(f"\n❌ Lỗi: {e}")


if __name__ == "__main__":
    main()

