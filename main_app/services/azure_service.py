# services/azure_service.py

import os
import json
import random
import string
import asyncio
from typing import Optional
from functools import lru_cache

from azure.storage.blob.aio import BlobServiceClient as AsyncBlobServiceClient
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
from core.config import settings

# --- HELPER FUNCTION FOR DIAGNOSTICS ---
def _log_storage_account_name(connection_string: str, client_type: str):
    """Safely extracts and prints the storage account name from a connection string for debugging."""
    try:
        parts = {p.split('=', 1)[0]: p.split('=', 1)[1] for p in connection_string.split(';')}
        account_name = parts.get('AccountName', 'N/A')
        print(f"--- DIAGNOSTIC ---: {client_type} is configured to use Azure Storage Account: '{account_name}'")
    except Exception:
        print("--- DIAGNOSTIC ---: Could not parse AccountName from connection string.")

# --- Async Client Initialization ---
@lru_cache(maxsize=1)
def _get_blob_service_client_async():
    if not settings.AZURE_CONNECTION_STRING:
        raise ValueError("FATAL: ASYNC AZURE_CONNECTION_STRING is not set.")
    # --- ADDED DIAGNOSTIC LOG ---
    _log_storage_account_name(settings.AZURE_CONNECTION_STRING, "Async Client (API)")
    return AsyncBlobServiceClient.from_connection_string(settings.AZURE_CONNECTION_STRING)

# --- Synchronous Client Initialization ---
@lru_cache(maxsize=1)
def _get_blob_service_client():
    if not settings.AZURE_CONNECTION_STRING:
        raise ValueError("FATAL: SYNC AZURE_CONNECTION_STRING is not set.")
    # --- ADDED DIAGNOSTIC LOG ---
    _log_storage_account_name(settings.AZURE_CONNECTION_STRING, "Sync Client (Celery)")
    return BlobServiceClient.from_connection_string(settings.AZURE_CONNECTION_STRING)


# --- (The rest of your azure_service.py file is unchanged) ---

# --- Asynchronous Download Function with Retry Logic ---
async def download_json_async(file_name: str, is_input: bool) -> Optional[dict]:
    container_name = settings.INPUT_JSON_CONTAINER if is_input else settings.OUTPUT_JSON_CONTAINER
    blob_service_client = _get_blob_service_client_async()
    max_retries = 3
    retry_delay_seconds = 0.5
    for attempt in range(max_retries):
        try:
            async with blob_service_client.get_container_client(container_name) as container_client:
                blob_client = container_client.get_blob_client(file_name)
                download_stream = await blob_client.download_blob()
                json_bytes = await download_stream.readall()
                return json.loads(json_bytes.decode('utf-8'))
        except ResourceNotFoundError:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay_seconds)
            else:
                return None
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while downloading '{file_name}': {e}")
            return None
    return None

# --- Synchronous Functions ---
def _get_container_client(container_name: str):
    blob_service_client = _get_blob_service_client()
    return blob_service_client.get_container_client(container_name)

def upload_json(data: dict, file_name: str, is_input: bool = False):
    container_name = settings.INPUT_JSON_CONTAINER if is_input else settings.OUTPUT_JSON_CONTAINER
    container_client = _get_container_client(container_name)
    json_bytes = json.dumps(data, indent=2).encode('utf-8')
    blob_client = container_client.get_blob_client(file_name)
    blob_client.upload_blob(json_bytes, overwrite=True)
    # print(f"Successfully uploaded JSON to {blob_client.url}") # We can comment this out to reduce noise

def save_image_locally_and_get_urls(image_bytes: bytes, file_name: str) -> dict:
    prefix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    sanitized_file_name = file_name.replace(" ", "_")
    final_filename = f"{prefix}_{sanitized_file_name}"
    storage_path = settings.LOCAL_IMAGE_STORAGE_PATH
    os.makedirs(storage_path, exist_ok=True)
    local_path = os.path.join(storage_path, final_filename)
    with open(local_path, "wb") as f:
        f.write(image_bytes)
    print(f"Image saved locally to: {local_path}")
    public_url = f"{settings.PUBLIC_IMAGE_BASE_URL}/{final_filename}"
    media_container_client = _get_container_client(settings.MEDIA_OUTPUT_CONTAINER)
    permanent_url = f"{media_container_client.url}/{final_filename}"
    return {"public_url": public_url, "permanent_url": permanent_url}