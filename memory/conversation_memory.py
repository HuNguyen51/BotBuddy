"""
Conversation Memory — Short-term memory cho agent.

Sử dụng LangGraph MemorySaver checkpointer - memory được quản lý
tự động bởi LangGraph thông qua thread_id.

Module này cung cấp thêm utility để:
- Lấy history từ checkpointer
- Format history cho context
"""

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()