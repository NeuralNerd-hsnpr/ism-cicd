#!/bin/bash

# Create logs directory
mkdir -p logs

echo "Starting Unified AI Services (Single Environment)..."

# 0. Start Redis (if not using external)
if [[ "$REDIS_URL" == *"localhost"* ]] || [[ -z "$REDIS_URL" ]]; then
    echo "Starting Local Redis..."
    redis-server --daemonize yes
fi

# 1. Start Flux Service (Port 8001)
echo "Starting Flux Service (8001)..."
cd /app/flux-fill-service
uvicorn app.main:app --host 0.0.0.0 --port 8001 > /app/logs/flux.log 2>&1 &

# 2. Start Lama Service (Port 8002)
echo "Starting Lama Service (8002)..."
cd /app/lama-service
uvicorn app.main:app --host 0.0.0.0 --port 8002 > /app/logs/lama.log 2>&1 &

# 3. Start RMBG Service (Port 8003)
echo "Starting RMBG Service (8003)..."
cd /app/rmbg-service
uvicorn app.main:app --host 0.0.0.0 --port 8003 > /app/logs/rmbg.log 2>&1 &

# 4. Start Gaze Service (Port 8004)
echo "Starting Gaze Service (8004)..."
cd /app/gaze-service
uvicorn app.main:app --host 0.0.0.0 --port 8004 > /app/logs/gaze.log 2>&1 &

# 5. Start Pix2Pix Service (Port 8005)
echo "Starting Pix2Pix Service (8005)..."
cd /app/stable-diffusion-pix2pix-service
uvicorn app.main:app --host 0.0.0.0 --port 8005 > /app/logs/pix2pix.log 2>&1 &

# 6. Start SDXL Service (Port 8006)
echo "Starting SDXL Service (8006)..."
cd /app/sdxl-inpainting-service
uvicorn app.main:app --host 0.0.0.0 --port 8006 > /app/logs/sdxl.log 2>&1 &

# 7. Start Wan Service (Port 8007)
echo "Starting Wan Service (8007)..."
cd /app/wan-service
uvicorn app.main:app --host 0.0.0.0 --port 8007 > /app/logs/wan.log 2>&1 &

# 8. Start Celery Workers
echo "Starting Celery Workers..."
cd /app/main_app
celery -A tasks.celery_app worker -Q inpainting_sync,inpainting_async,treatment_sync,treatment_async --loglevel=info -n worker_inpainting@%h > /app/logs/worker_inpainting.log 2>&1 &
celery -A tasks.celery_app worker -Q background_sync,background_async --loglevel=info -n worker_background@%h > /app/logs/worker_background.log 2>&1 &
celery -A tasks.celery_app worker -Q video_async --loglevel=info -n worker_video@%h > /app/logs/worker_video.log 2>&1 &

# 9. Start Main App (Port 8000)
echo "Starting Main App (8000)..."
cd /app/main_app
uvicorn app.main:app --host 0.0.0.0 --port 8000 > /app/logs/main_app.log 2>&1 &

echo "All services started. Tailing logs..."
tail -f /app/logs/*.log
