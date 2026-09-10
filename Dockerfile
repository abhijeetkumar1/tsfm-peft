# CPU by default so `docker build .` works anywhere and in CI.
# For a CUDA image:
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 -t tsfm-peft:cu124 .
ARG PYTHON_VERSION=3.11
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    HF_HOME=/cache/huggingface \
    TSFM_PEFT_CACHE=/cache/tsfm_peft

WORKDIR /app

# Torch first, from its own index, in its own layer: it is by far the largest
# dependency and it should not be re-downloaded when anything else changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --index-url "${TORCH_INDEX_URL}" "torch>=2.4"

# Then the package and its remaining deps. src/ is needed because the build
# backend reads it; configs/scripts/tests are copied afterwards so that editing
# an experiment config does not invalidate the dependency layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system ".[models,dev]"

COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests

# Datasets and model weights are downloaded at runtime into /cache, which is
# intended to be a mounted volume. Nothing is baked into the image.
RUN useradd --create-home --uid 1000 runner \
    && mkdir -p /cache \
    && chown -R runner:runner /cache /app
USER runner
VOLUME ["/cache"]

ENTRYPOINT ["python", "-m", "tsfm_peft.cli"]
CMD ["--help"]
