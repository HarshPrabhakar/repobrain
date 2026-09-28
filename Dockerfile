FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    REPOBRAIN_INDEX_DIR=/data/indexes HF_HOME=/data/models \
    REPOBRAIN_EMBEDDING_DEVICE=cpu
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml readme.md ./
COPY repobrain ./repobrain
RUN pip install --no-cache-dir .
RUN useradd --create-home --uid 10001 repobrain && mkdir -p /data/indexes /data/models && chown -R repobrain:repobrain /data
USER repobrain
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["python", "-m", "uvicorn", "repobrain.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
