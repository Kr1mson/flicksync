FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv venv /app/.venv && \
    uv sync --frozen --no-dev && \
    pip uninstall -y uv

ENV PATH="/app/.venv/bin:$PATH"

COPY utility/ utility/
COPY app.py .
COPY frontend/ frontend/

RUN adduser --disabled-password --gecos "" appuser
USER appuser

VOLUME ["/app/embeddings"]
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]