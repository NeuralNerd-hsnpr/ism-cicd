# Use a CUDA-enabled base image for production.
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
# For GPU, you might need a specific torch install command here
# e.g., RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app/app

# Expose the port the app will run on (e.g., 8006)
EXPOSE 8006

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8006"]