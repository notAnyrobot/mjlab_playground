# Refer to uv-docker-example:
# https://github.com/astral-sh/uv-docker-example/blob/main/standalone.Dockerfile
# This is a development/training image. The heavy dependency environment is
# baked into /app/.venv, while launchers mount the live checkout elsewhere.

FROM nvcr.io/nvidia/cuda:12.8.0-runtime-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libegl-dev \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_PREFERENCE=only-managed
ENV UV_HTTP_TIMEOUT=300

RUN uv python install 3.13

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable --no-dev

ADD . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --no-dev

ENV PATH=/app/.venv/bin:$PATH
ENV MUJOCO_GL=egl
ENV UV_NO_SYNC=1
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV VIRTUAL_ENV=/app/.venv

EXPOSE 8080

CMD ["bash"]
