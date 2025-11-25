# It is highly recommended to use an official NVIDIA CUDA base image for this service.
# Example: FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
# For GPU, you MUST install the correct PyTorch version for your CUDA driver
# Example: RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app/app

# Expose the port the app will run on (e.g., 8011)
EXPOSE 8011

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8011"]