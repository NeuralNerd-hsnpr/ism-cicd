# your_project/worker.py

from celery import Celery
from kombu import Queue
from core.config import settings

celery_app = Celery(
    "worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    # --- MODIFIED: Add the new task module to the include list ---
    include=[
        'tasks.inpainting_task',
        'tasks.treatment_task',
        'tasks.background_task',
        'tasks.video_task'
    ]
)

# --- MODIFIED: Add new queues for the treatment workload ---
celery_app.conf.task_queues = (
    Queue("inpainting_async"),
    Queue("inpainting_sync"),
    Queue("treatment_async"),
    Queue("treatment_sync"),
    Queue("background_async"),
    Queue("background_sync"),
    Queue("video_async"),
)

# --- (Standard Celery Configuration - Unchanged) ---
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
)