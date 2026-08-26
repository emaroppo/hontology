# Image for the optional ingest watcher. The API and UI are normally run from a
# local venv during development; this exists so unattended ingest can be a
# long-running container with a restart policy.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer first, so source edits do not reinstall the world.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Run as a non-root user; nothing here needs privileges.
RUN useradd --create-home --uid 10001 hontology \
    && mkdir -p /data \
    && chown -R hontology:hontology /app /data
USER hontology

ENTRYPOINT []
CMD ["hontology", "ingest", "status"]
