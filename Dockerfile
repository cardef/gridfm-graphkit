# syntax=docker/dockerfile:1
FROM docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

LABEL org.opencontainers.image.source="https://github.com/cardef/gridfm-graphkit" \
      org.opencontainers.image.description="gridfm-graphkit" \
      org.opencontainers.image.version="0.8.1"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    HOME=/tmp \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && python -m ensurepip --upgrade \
    && python -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# Pin the container runtime while leaving the Python package's support range open.
RUN python -m pip install --no-cache-dir \
        "torch==2.10.*" \
        --index-url "${PYTORCH_INDEX_URL}"

COPY pyproject.toml README.md LICENSE ./
COPY gridfm_graphkit ./gridfm_graphkit

RUN python -m pip install --no-cache-dir \
        --extra-index-url "${PYTORCH_INDEX_URL}" \
        . \
    && chgrp -R 0 /app \
    && chmod -R g=u /app

USER 1001

ENTRYPOINT ["gridfm_graphkit"]
CMD ["--help"]
