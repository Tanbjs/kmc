FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git && \
    rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install -e .

COPY sysid/requirements.txt ./sysid/requirements.txt
RUN pip install -r sysid/requirements.txt

CMD ["bash"]
