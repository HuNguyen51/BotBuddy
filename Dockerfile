FROM python:3.13-slim

RUN pip install uv --no-cache-dir

COPY . /app

WORKDIR /app

RUN uv pip install . --system --no-cache-dir

RUN mkdir -p logs data/chroma_db data/documents

CMD ["uv", "run", "python", "main.py"]