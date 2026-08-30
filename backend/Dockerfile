# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS builder

WORKDIR /build
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN python -m venv "${VIRTUAL_ENV}"
COPY requirements-prod.txt ./
# Render's build sandbox does not permit swapon. Keep pip's cache outside the
# image and use a paid/pre-built build environment when memory is constrained.
RUN --mount=type=cache,target=/root/.cache/pip \
	pip install --no-cache-dir --upgrade pip \
	&& pip install --no-cache-dir -r requirements-prod.txt

FROM python:3.11-slim AS runtime

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:${PATH}"

COPY --from=builder /opt/venv /opt/venv
COPY . ./

# model.safetensors is intentionally copied from the build context. Download
# it before building (or use a registry image) instead of downloading in Docker.
RUN useradd --create-home appuser \
	&& chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
