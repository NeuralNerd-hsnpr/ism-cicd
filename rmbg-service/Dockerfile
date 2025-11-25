# Use a CUDA-enabled base image for production, or a standard Python image for CPU.
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
# If using GPU, you might need a specific torch install command here
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./app /app/app

# Expose the port the app will run on (e.g., 8007)
EXPOSE 8003

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]