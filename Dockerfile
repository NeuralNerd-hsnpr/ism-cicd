# Unified Dockerfile for All AI Services
# Uses a single stable environment as requested

FROM python:3.11-slim

# 1. Install System Dependencies
# libgl1/libglib2.0-0 are for OpenCV
# git is for installing from git repos
# redis-server is for local redis if needed
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    redis-server \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Set working directory
WORKDIR /app

# 3. Install Python Dependencies
# We copy the unified requirements.txt first to leverage Docker cache
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy Codebase
COPY . /app

# 5. Make entrypoint executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# 6. Expose all ports
EXPOSE 8000 8001 8002 8003 8004 8005 8006 8007

# 7. Start Services
CMD ["/app/entrypoint.sh"]
