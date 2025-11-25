import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class Settings:
    # --- Project Directory Setup ---
    # BASE_DIR will be the absolute path to your project's root folder.
    # This is a robust way to handle paths regardless of where the script is run.
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # LOCAL_IMAGE_STORAGE_PATH is now defined relative to the project root.
    # This will create a folder named 'local_image_storage' inside your main project directory.
    LOCAL_IMAGE_STORAGE_PATH: Path = BASE_DIR / "local_image_storage"

    # --- Redis Configuration ---
    # Get Redis URL from environment variables for Celery broker and backend
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # --- Azure Configuration ---
    # Connection string is loaded from environment variables
    AZURE_CONNECTION_STRING: str = os.getenv("AZURE_CONNECTION_STRING")
    
    #Azure Containers Setting
    INPUT_JSON_CONTAINER: str = "modification-inputs-json"
    OUTPUT_JSON_CONTAINER: str = "modification-outputs-json"
    MEDIA_OUTPUT_CONTAINER: str = "modification-outputs-media"

    # --- Service URLs ---
    # The URL for the downstream Flux inference service
    # Defaulting to 8001, but allowing override.
    INFERENCE_SERVICE_URL: str = os.getenv("INFERENCE_SERVICE_URL", "http://localhost:8001/inpaint")

    # The public-facing base URL for serving locally saved images
    PUBLIC_IMAGE_BASE_URL: str = os.getenv("PUBLIC_IMAGE_BASE_URL", "http://localhost:8000/images")

    LAMA_SERVICE_URL: str = os.getenv("LAMA_SERVICE_URL", "http://localhost:8002/remove")
    RMBG_SERVICE_URL: str = os.getenv("RMBG_SERVICE_URL", "http://localhost:8003/remove_background")
    GAZE_SERVICE_URL: str = os.getenv("GAZE_SERVICE_URL", "http://localhost:8004/get_direction_by_mask")
    DETECTION_SERVICE_URL: str = os.getenv("DETECTION_SERVICE_URL", "http://173.208.151.202:8000/v1_0/image-processing/object/detect")
    PIX2PIX_SERVICE_URL: str = os.getenv("PIX2PIX_SERVICE_URL", "http://localhost:8005/style")
    SDXL_INPAINTING_SERVICE_URL: str = os.getenv("SDXL_INPAINTING_SERVICE_URL", "http://localhost:8006/inpaint") # Port 8006
    WAN_SERVICE_URL: str = os.getenv("WAN_SERVICE_URL", "http://localhost:8007/generate_video") # Port 8007

    


# Create a single, importable instance of the settings
settings = Settings()