FROM python:3.11-slim

RUN useradd -m -u 1000 user

# Install PyTorch (CPU) + core dependencies
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    jupyterlab>=4.0 \
    numpy \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    python-multipart>=0.0.6

WORKDIR /app

# Install torch_judge package
COPY torch_judge/ ./torch_judge/
COPY setup.py ./
RUN pip install --no-cache-dir -e .

# Copy application code
COPY web/ ./web/
COPY start_web.py start_jupyter.py ./
COPY templates/ ./templates/
COPY solutions/ ./solutions/
COPY entrypoint.sh ./
RUN chmod +x /app/entrypoint.sh

# Create data directories and set ownership
RUN mkdir -p /app/notebooks /app/data && \
    chown -R user:user /app

USER user

ENV MODE=web
EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
