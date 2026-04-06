ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY content_moderation_env/ /app/content_moderation_env/

WORKDIR /app/content_moderation_env

RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

RUN uv pip install --system \
    "openenv-core[core]>=0.2.2" \
    "uvicorn>=0.24.0" \
    "fastapi>=0.104.0"

FROM ${BASE_IMAGE}

COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY content_moderation_env/ /app/content_moderation_env/

ENV PYTHONPATH="/app:$PYTHONPATH"

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "content_moderation_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]