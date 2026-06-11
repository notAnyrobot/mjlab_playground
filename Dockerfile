# Refer to uv-docker-example:
# https://github.com/astral-sh/uv-docker-example/blob/main/standalone.Dockerfile
# This is a development/training image. The heavy dependency environment is
# baked into /app/.venv, while launchers mount the live checkouts elsewhere.

FROM nvcr.io/nvidia/cuda:12.8.0-runtime-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /uvx /bin/

ARG PIP_INDEX_URL=
ARG APT_MIRROR=
ARG UV_PYTHON_INSTALL_MIRROR=
ARG UV_HTTP_TIMEOUT=3600
ARG UV_CONCURRENT_DOWNLOADS=1

ENV DEBIAN_FRONTEND=noninteractive
RUN set -eu; \
    if [ -n "$APT_MIRROR" ]; then \
      apt_mirror="${APT_MIRROR%/}/"; \
      for sources in /etc/apt/sources.list /etc/apt/sources.list.d/*.sources; do \
        [ -e "$sources" ] || continue; \
        sed -i \
          -e "s|http://archive.ubuntu.com/ubuntu/|$apt_mirror|g" \
          -e "s|http://security.ubuntu.com/ubuntu/|$apt_mirror|g" \
          "$sources"; \
      done; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      git \
      curl \
      libegl-dev; \
    rm -rf /var/lib/apt/lists/*

COPY scripts/docker_uv_sync_locked.sh /usr/local/bin/uv-sync-locked

RUN chmod +x /usr/local/bin/uv-sync-locked

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_PREFERENCE=only-managed
ENV UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT}
ENV UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS}
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV VIRTUAL_ENV=/app/.venv

RUN --mount=type=cache,target=/root/.cache/uv \
    rm -rf /root/.local/share/uv/python/.temp; \
    if [ -n "$UV_PYTHON_INSTALL_MIRROR" ]; then export UV_PYTHON_INSTALL_MIRROR; fi; \
    uv python install 3.13

ENV GIT_HTTP_LOW_SPEED_LIMIT=1
ENV GIT_HTTP_LOW_SPEED_TIME=600

RUN git config --global http.version HTTP/1.1

WORKDIR /workspace/mujocolab/mjlab_playground

RUN mkdir -p /workspace/mujocolab/mjlab

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,from=mjlab,source=pyproject.toml,target=/workspace/mujocolab/mjlab/pyproject.toml \
    uv-sync-locked --no-install-workspace --no-editable --no-dev

COPY --from=mjlab . /workspace/mujocolab/mjlab
ADD . /workspace/mujocolab/mjlab_playground

RUN --mount=type=cache,target=/root/.cache/uv \
    uv-sync-locked --no-editable --no-dev

ENV PATH=/app/.venv/bin:$PATH
ENV MUJOCO_GL=egl
ENV UV_NO_SYNC=1

EXPOSE 8080

CMD ["bash"]
