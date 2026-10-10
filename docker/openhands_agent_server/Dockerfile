# Multi-stage build for lightweight standalone openhands-agent-server (~200MB)
# Pinned to OpenHands 0.62 ecosystem (openhands-agent-server==1.0.0a6, openhands-sdk==1.0.0a6)
FROM python:3.12-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    patchelf \
    binutils \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    'openhands-agent-server==1.0.0a6' \
    'openhands-sdk==1.0.0a6' \
    'openhands-tools==1.0.0a6' \
    'openhands-aci==0.3.2' \
    'pyinstaller>=6.0.0'

WORKDIR /build
COPY entrypoint.py .

RUN pyinstaller --noconfirm --onefile --clean \
    --name openhands-agent-server \
    --collect-all openhands \
    --collect-all litellm \
    --collect-all tiktoken \
    --collect-all tiktoken_ext \
    --collect-all fastmcp \
    --collect-all mcp \
    --collect-all uvicorn \
    --collect-all fastapi \
    --collect-all starlette \
    --collect-all pydantic \
    entrypoint.py

RUN mkdir -p /opt/oh/glibc236 && \
    cp dist/openhands-agent-server /opt/oh/openhands-agent-server && \
    cp /lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 /opt/oh/glibc236/ && \
    cp /lib/x86_64-linux-gnu/libc.so.6 /opt/oh/glibc236/ && \
    cp /lib/x86_64-linux-gnu/libm.so.6 /opt/oh/glibc236/ && \
    (cp /lib/x86_64-linux-gnu/libdl.so.2 /opt/oh/glibc236/ 2>/dev/null || true) && \
    (cp /lib/x86_64-linux-gnu/libpthread.so.0 /opt/oh/glibc236/ 2>/dev/null || true) && \
    (cp /lib/x86_64-linux-gnu/librt.so.1 /opt/oh/glibc236/ 2>/dev/null || true) && \
    patchelf --set-interpreter /oh/glibc236/ld-linux-x86-64.so.2 /opt/oh/openhands-agent-server && \
    patchelf --set-rpath /oh/glibc236 /opt/oh/openhands-agent-server && \
    chmod +x /opt/oh/openhands-agent-server

FROM debian:bookworm-slim
COPY --from=builder /opt/oh /opt/oh
CMD ["sh", "-c", "cp -a /opt/oh/* /oh/"]
