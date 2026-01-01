# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /workspace


FROM python:3.10-slim

# 1) Install sudo and other packages as root
RUN apt-get update && \
    apt-get install -y sudo && \
    rm -rf /var/lib/apt/lists/*

# 2) Create non-root user (still as root)
RUN useradd -m -s /bin/bash devuser && \
    echo "devuser:ChangeMe123" | chpasswd && \
    adduser devuser sudo

# 3) Copy everything into /workspace (as root)
WORKDIR /workspace
COPY . /workspace

# 4) Clean up all Docker-/build-files (still as root)
RUN rm -f /workspace/Dockerfile \
          /workspace/.dockerignore \
          /workspace/*.sh \
          /workspace/*.yml \
          /workspace/LICENSE.md \
          /workspace/README.md

# 5) (Optional) chown so devuser can write to /workspace if needed
RUN chown -R devuser:devuser /workspace

# 6) Switch to non-root user
USER devuser

# 7) Launch a login shell for the $ prompt
CMD ["bash", "-l"]
