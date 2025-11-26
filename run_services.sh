#!/bin/bash

# Create logs directory
mkdir -p logs

# Function to kill all background jobs on exit
cleanup() {
    echo ""
    echo "Stopping all services..."
    kill $(jobs -p) 2>/dev/null
    echo "All services stopped."
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM EXIT

echo "Starting Microservices Ecosystem (Background Mode)..."
ROOT_DIR=$(pwd)
echo "Root Directory: $ROOT_DIR"
mkdir -p "$ROOT_DIR/logs"
echo "Logs will be written to $ROOT_DIR/logs/"

# 0. Start Red
echo "Starting Redis..."
if command -v redis-server &> /dev/null; then
    redis-server > logs/redis.log 2>&1 &
else
    echo "redis-server not found. Attempting to start via Docker..."
    if docker ps | grep -q redis; then
        echo "Redis container is already running."
    elif docker ps -a | grep -q redis; then
        echo "Starting existing Redis container..."
        docker start redis
    else
        echo "Starting new Redis container..."
        docker run -d --name redis -p 6379:6379 redis:alpine
    fi
fi 

# 1. Start Flux Service (Port 8001)
echo "Starting Flux Service (8001)..." 
(cd "$ROOT_DIR/flux-fill-service" && uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload) > "$ROOT_DIR/logs/flux.log" 2>&1 &

# 2. Start Lama Service (Port 8002)
echo "Starting Lama Service (8002)..."
(cd "$ROOT_DIR/lama-service" && uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload) > "$ROOT_DIR/logs/lama.log" 2>&1 &

# 3. Start RMBG Service (Port 8003)
echo "Starting RMBG Service (8003)..."
(cd "$ROOT_DIR/rmbg-service" && uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload) > "$ROOT_DIR/logs/rmbg.log" 2>&1 &

# 4. Start Gaze Service (Port 8004)
echo "Starting Gaze Service (8004)..."
(cd "$ROOT_DIR/gaze-service" && uvicorn app.main:app --host 0.0.0.0 --port 8004 --reload) > "$ROOT_DIR/logs/gaze.log" 2>&1 &

# 5. Start Pix2Pix Service (Port 8005)
echo "Starting Pix2Pix Service (8005)..."
(cd "$ROOT_DIR/stable-diffusion-pix2pix-service" && uvicorn app.main:app --host 0.0.0.0 --port 8005 --reload) > "$ROOT_DIR/logs/pix2pix.log" 2>&1 &

# 6. Start SDXL Inpainting Service (Port 8006)
echo "Starting SDXL Service (8006)..."
(cd "$ROOT_DIR/sdxl-inpainting-service" && uvicorn app.main:app --host 0.0.0.0 --port 8006 --reload) > "$ROOT_DIR/logs/sdxl.log" 2>&1 &

# 7. Start Wan Video Service (Port 8007)
# echo "Starting Wan Service (8007)..."
# (cd "$ROOT_DIR/wan-service" && uvicorn app.main:app --host 0.0.0.0 --port 8007 --reload) > "$ROOT_DIR/logs/wan.log" 2>&1 &

# 8. Start Main App (Port 8000)
echo "Starting Main App (8000)..."
(cd "$ROOT_DIR/main_app" && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload) > "$ROOT_DIR/logs/main_app.log" 2>&1 &

# 9. Start Celery Worker 1 (Inpainting)
echo "Starting Celery Worker: Inpainting..."
(cd "$ROOT_DIR/main_app" && celery -A tasks.celery_app worker -Q inpainting_sync,inpainting_async,treatment_sync,treatment_async --loglevel=info -n worker_inpainting@%h) > "$ROOT_DIR/logs/worker_inpainting.log" 2>&1 &

# 10. Start Celery Worker 2 (Background)
echo "Starting Celery Worker: Background..."
(cd "$ROOT_DIR/main_app" && celery -A tasks.celery_app worker -Q background_sync,background_async --loglevel=info -n worker_background@%h) > "$ROOT_DIR/logs/worker_background.log" 2>&1 &

# 11. Start Celery Worker 3 (Video)
# echo "Starting Celery Worker: Video..."
# (cd "$ROOT_DIR/main-app" && celery -A tasks.celery_app worker -Q video_async --loglevel=info -n worker_video@%h) > "$ROOT_DIR/logs/worker_video.log" 2>&1 &

echo "---------------------------------------------------"
echo "All services started."
echo "Main App: http://localhost:8000"
echo "Logs are in ./logs/"
echo "Press Ctrl+C to stop all services."
echo "---------------------------------------------------"

# Keep script running and show logs
echo "Tailing logs (Ctrl+C to stop)..."
tail -f logs/*.log
