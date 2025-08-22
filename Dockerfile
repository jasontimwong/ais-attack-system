# AIS Attack Generation System - Docker Image
# Multi-stage build for optimized production image

# Stage 1: Build environment
FROM python:3.9-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libgeos-dev \
    libproj-dev \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# Stage 2: Production environment
FROM python:3.9-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libgeos-c1v5 \
    libproj15 \
    libgdal26 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r aisattack && \
    useradd -r -g aisattack -d /app -s /bin/bash aisattack

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=aisattack:aisattack . .

# Create necessary directories
RUN mkdir -p /app/data /app/output /app/logs && \
    chown -R aisattack:aisattack /app

# Switch to non-root user
USER aisattack

# Install the package in development mode
RUN pip install -e .

# Expose ports
EXPOSE 8000 5173

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python tools/system_check.py --quiet || exit 1

# Default command
CMD ["python", "-m", "core.cli", "--help"]

# Stage 3: Development environment
FROM production as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
COPY requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install -r /tmp/requirements-dev.txt

# Install pre-commit hooks
RUN pip install pre-commit

# Switch back to aisattack user
USER aisattack

# Development command
CMD ["bash"]

# Stage 4: Web interface build
FROM node:16-alpine as web-builder

WORKDIR /app/web

# Copy package files
COPY visualization/web_interface/package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY visualization/web_interface/ .

# Build the application
RUN npm run build

# Stage 5: Full system with web interface
FROM production as full-system

# Copy web build from web-builder stage
COPY --from=web-builder /app/web/dist /app/visualization/web_interface/dist

# Install Node.js for serving web interface
USER root
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g serve

USER aisattack

# Expose additional port for web interface
EXPOSE 3000

# Start script for full system
COPY --chown=aisattack:aisattack docker/start-full-system.sh /app/start-full-system.sh
RUN chmod +x /app/start-full-system.sh

CMD ["/app/start-full-system.sh"]
