# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional dependencies for production
RUN pip3 install --no-cache-dir \
    prometheus-client \
    gputil \
    fastapi \
    uvicorn \
    celery \
    redis \
    psutil

# Copy application code
COPY agent_company/ ./agent_company/

# Create necessary directories
RUN mkdir -p /app/data /app/checkpoints /app/logs /app/cache

# Set permissions
RUN chmod +x /app/agent_company/*.py

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python3", "-m", "agent_company.enhanced_pipeline", "--environment", "production"] 