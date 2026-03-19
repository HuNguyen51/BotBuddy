# ScanX Agent

<p align="center">
  <em>An elegant, multi-tenant AI Assistant API designed primarily for F&B customer service, providing hyper-personalized recommendations, menu search, and real-time streaming capabilities.</em>
</p>

---

## 📖 Overview

**ScanX Agent** is a sophisticated AI-powered customer service backend built with **FastAPI**, **LangGraph**, and **LangChain**. It focuses on delivering contextual, persona-driven, and multi-tenant conversational experiences. 

Currently optimized as an **F&B (Food & Beverage) Agent**, it can seamlessly query menus, fetch product details, and recommend dishes dynamically based on the customer's queries and context while supporting **RAG (Retrieval-Augmented Generation)** through **Qdrant** and **Voyage AI** embeddings.

## ✨ Key Features

- **Multi-Tenant Architecture**: Supports isolated contexts for different tenants (e.g., various restaurants or brands) by injecting `tenant_id` into tools under the hood without exposing it to the LLM.
- **RAG-Powered Conversations**: Integrates a **Qdrant** vector database and **Voyage4NanoEmbedding** to retrieve real-time menu items and contextual data.
- **Dynamic AI Agent (LangGraph)**: Stateful conversational agents capable of advanced tool calling and reasoning based on LangGraph orchestration.
- **Unified LLM Interface**: Implements **LiteLLM** to seamlessly switch and route between multiple providers (OpenAI, Google Gemini 2.5 Flash, Local Ollama models) without altering business logic.
- **Real-time Streaming**: Supports Server-Sent Events (SSE) via FastAPI to stream LLM responses back to the client instantly.
- **Rich Toolset Built-in**: 
  - 🍽️ `menu_search`: Semantic search for menus and dishes.
  - ℹ️ `get_product_detail`: Fetch specific product details.
  - 💡 `get_recommendations`: Intelligent dish upselling and recommendations.
  - 🌐 `web_search`: General duckduckgo web search.
  - 🧮 `calculator`: Basic mathematical reasoning support.
- **Observability**: Centralized logging, retry mechanisms, and tracing configurations (via Langfuse).

## 🛠️ Technology Stack

- **Framework**: FastAPI, Uvicorn (SSE Starlette for streaming)
- **AI/LLM Ecosystem**: LangChain, LangGraph, LiteLLM
- **Vector Database**: Qdrant (Local / In-memory or Cloud)
- **Embeddings**: Voyage AI (`voyage-4-nano`), Sentence-Transformers
- **Package Management**: `uv` (Modern Python package installer & resolver)
- **Configuration & Validation**: Pydantic v2, Python-dotenv

## 📂 Project Structure

```text
scanx-agent/
├── agents/             # LangGraph-based AI Agents (BaseAgent, FnBAgent, ResearchAgent)
├── configs/            # Application settings and model configurations
├── data/               # Vector Store (Qdrant) logic, Embeddings, and Data Ingestion
├── endpoints/          # FastAPI Routers (/chat, /chat/stream schemas)
├── services/           # Core Services (LLM inference wrapped via LiteLLM)
├── tools/              # LangChain-compatible Agent Tools
├── memory/             # Agent Memory & Context Checkpointing
├── utils/              # Shared Utilities (Logger, Retry, Tracing)
├── external_data_storage/ # Static/JSON knowledge base files (e.g., fnb.json)
├── tests/              # Pytest unittests and async testing
├── server.py           # Main FastAPI Application entry point
├── pyproject.toml      # Project Metadata & Dependencies
└── uv.lock             # Completely locked dependencies for reproducible environments
```

## 🚀 Getting Started

### 1. Prerequisites
- Python `>= 3.11`
- `uv` package manager installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

### 2. Installation
Clone the repository and install the dependencies:
```bash
# Sync dependencies and create a virtual environment automagically
uv sync
```

### 3. Environment Configuration
Create an environment file:
```bash
cp .env.example .env
```
Ensure you configure your respective API keys inside `.env` (e.g., `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `VOYAGE_API_KEY`, `QDRANT_URL`).

### 4. Data Ingestion (Vector Store)
Before running the F&B agent, you must ingest the mock data into the Qdrant vector store.
```bash
# Run the ingestion script (Ensure external_data_storage/fnb.json exists)
uv run python -m data.ingestion.fnb_ingestor
```

### 5. Start the API Server
Run the FastAPI application via `uvicorn`:
```bash
uv run uvicorn server:app --reload --port 8000
```
*The server will start at `http://localhost:8000`.*

---

## 📡 API Endpoints

- **`GET /health`**: Health Check API.
- **`GET /docs`**: OpenAPI Swagger UI interface.
- **`POST /chat`**: Synchrnous full-response chat endpoint.
- **`POST /chat/stream`**: Real-time SSE streaming endpoint for interactive frontend experiences.

**Sample Request (`/chat`):**
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
           "message": "Tôi muốn ăn đồ gì đó cay cay và uống bia",
           "session_id": "session-1234",
           "tenant_id": "tenant-abc"
         }'
```

## 🗺️ Roadmap & Development Status
- [x] Integrate LangGraph agent brains & custom tools.
- [x] Expose Server-Sent Events (SSE) Streaming APIs.
- [x] Configure Qdrant RAG + Voyage AI embeddings.
- [x] LLM Multi-provider Routing (LiteLLM).
- [ ] Implement robust persistent Memory Saving mechanisms.
- [ ] Connect with structured Customer CRM Data (Order history).
- [ ] Invoice creation and autonomous Payment flows.

---

<p align="center">
  Developed by the <b>ScanX Team</b>
</p>
