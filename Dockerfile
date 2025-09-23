FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Set working directory
WORKDIR /app

# Copy uv configuration files
COPY pyproject.toml uv.lock ./

# Install dependencies with uv
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/

# Create required directories for persistent data
RUN mkdir -p /app/data/vector_store \
    && mkdir -p /app/data/document_store \
    && chmod -R 755 /app/data

# Expose Streamlit port
EXPOSE 8501

# Environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Startup command
CMD ["uv", "run", "streamlit", "run", "src/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
