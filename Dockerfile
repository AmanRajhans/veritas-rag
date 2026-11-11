FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VERITAS_INDEX_PATH=/app/artifacts/demo-index.json

WORKDIR /app
RUN useradd --create-home --uid 10001 appuser

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY data ./data
RUN mkdir -p artifacts && \
    veritas-rag index data/demo --output artifacts/demo-index.json && \
    chown -R appuser:appuser /app

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"

CMD ["uvicorn", "veritas_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
