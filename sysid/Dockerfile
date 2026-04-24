# Dockerfile
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git
RUN python3 -m pip install --upgrade pip setuptools wheel

# working directory
WORKDIR /workspace

# install dependencies
COPY requirements/ ./requirements/
RUN pip install -r requirements/base.txt
RUN pip install -r requirements/dev.txt

CMD ["bash"]