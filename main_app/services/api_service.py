# services/ai_clients.py

import httpx
import random
from core.config import settings
from services import azure_service

# --- Service URLs from your central settings ---
# Defined in core/config.py

class AIRequestError(Exception):
    """Custom exception for all AI service failures."""
    pass

def call_detection_service(parent_id: str, image_id: str, image_bytes: bytes, detect_phrase: str) -> str:
    """Calls the detection microservice and returns the resulting mask URL."""
    print(f"AI_CLIENT: Calling detection service for '{detect_phrase}'")
    detection_input_url = azure_service.upload_image_and_get_public_url(
        image_bytes=image_bytes,
        file_name=f"{parent_id}_{image_id}_detection_input.png"
    )
    payload = {
        "parentID": f"{parent_id}(dynamic_detection)", "method": "sync",
        "images": [{"imageID": image_id, "url": detection_input_url, "detect": [detect_phrase]}],
        "actions": ["OBJECT_DETECTION"], "returns": ["MASK_IMAGE"],
    }
    try:
        # NOTE: You should add DETECTION_SERVICE_URL to your settings file if you want to parameterize this.
        # Currently hardcoded as per previous implementation.
        with httpx.Client(timeout=120.0) as client:
            response = client.post(settings.DETECTION_SERVICE_URL, json=payload)
            response.raise_for_status()
            result = response.json()
        objects = result["output"]["imageResults"][0]["objects"]
        if not objects:
            raise AIRequestError(f"Detection service failed to detect '{detect_phrase}'.")
        return random.choice(objects)["maskTemporary"]
    except httpx.HTTPStatusError as e:
        raise AIRequestError(f"Detection service failed with status {e.response.status_code}: {e.response.text}") from e
    except (KeyError, IndexError) as e:
        raise AIRequestError(f"Failed to parse response from detection service: {e}") from e

def call_lama_service(image_bytes: bytes, mask_bytes: bytes) -> bytes:
    """Calls the object removal (Lama) microservice."""
    print("AI_CLIENT: Calling LAMA removal service.")
    files = {
        'image': ('image.png', image_bytes, 'image/png'),
        'mask': ('mask.png', mask_bytes, 'image/png')
    }
    try:
        with httpx.Client(timeout=300.0) as client:
            response = client.post(settings.LAMA_SERVICE_URL, files=files)
            response.raise_for_status()
            return response.content
    except httpx.HTTPStatusError as e:
        raise AIRequestError(f"Lama removal service failed with status {e.response.status_code}: {e.response.text}") from e
    except httpx.RequestError as e:
        raise AIRequestError(f"Could not connect to Lama removal service: {e}") from e

def call_flux_service(image_bytes: bytes, mask_bytes: bytes, prompt: str, params: dict = None) -> bytes:
    """Calls the Flux Inpainting microservice."""
    print(f"AI_CLIENT: Calling Flux Inpainting service for prompt '{prompt}'")
    files_payload = {
        'base_image': ('base_image.png', image_bytes, 'image/png'),
        'mask_image': ('mask_image.png', mask_bytes, 'image/png')
    }
    data_payload = {'prompt': prompt}
    if params:
        data_payload.update(params)
    data_payload = {k: v for k, v in data_payload.items() if v is not None}
    try:
        with httpx.Client(timeout=500.0) as client:
            response = client.post(settings.INFERENCE_SERVICE_URL, files=files_payload, data=data_payload)
            response.raise_for_status()
            return response.content
    except httpx.HTTPStatusError as e:
        raise AIRequestError(f"Flux Inpainting service failed with status {e.response.status_code}: {e.response.text}") from e
    except httpx.RequestError as e:
        raise AIRequestError(f"Could not connect to Flux Inpainting service: {e}") from e

def call_gaze_service(base_image_bytes: bytes, mask_image_bytes: bytes) -> str:
    """
    Calls the Gaze Estimation Service to determine the head pose (left/right/center).
    """
    print("AI_CLIENT: Calling Gaze Estimation service.")
    url = settings.GAZE_SERVICE_URL
    files = {
        "base_image": ("base_image.png", base_image_bytes, "image/png"),
        "face_mask": ("face_mask.png", mask_image_bytes, "image/png")
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, files=files)
            response.raise_for_status()
            result = response.json()
            return result.get("direction", "center")
            
    except httpx.HTTPStatusError as e:
        raise AIRequestError(f"Gaze service error: {e.response.text}", service_name="Gaze", status_code=e.response.status_code)
    except httpx.RequestError as e:
        raise AIRequestError(f"Gaze service connection failed: {str(e)}", service_name="Gaze")
    except Exception as e:
        raise AIRequestError(f"Unexpected error calling Gaze service: {str(e)}", service_name="Gaze")

    except httpx.RequestError as e:
        raise AIRequestError(f"Gaze service connection failed: {str(e)}", service_name="Gaze")
    except Exception as e:
        raise AIRequestError(f"Unexpected error calling Gaze service: {str(e)}", service_name="Gaze")

def call_style_service(image_bytes: bytes, style_prompt: str, model: str = "controlnet") -> bytes:
    """
    Calls the Stable Diffusion Styling microservice.
    Supports 'pix2pix' and 'controlnet'.
    """
    print(f"AI_CLIENT: Calling Style service with model '{model}'.")
    url = settings.PIX2PIX_SERVICE_URL
    files = {"image": ("image.png", image_bytes, "image/png")}
    data = {
        "prompt": style_prompt,
        "model": model,
        "num_inference_steps": 50,
        "image_guidance_scale": 1.5
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, files=files, data=data)
            response.raise_for_status()
            return response.content
    except httpx.HTTPStatusError as e:
        raise AIRequestError(f"Style service error: {e.response.text}", service_name="Style", status_code=e.response.status_code)
    except httpx.RequestError as e:
        raise AIRequestError(f"Style service connection failed: {str(e)}", service_name="Style")
    except Exception as e:
        raise AIRequestError(f"Unexpected error calling Style service: {str(e)}", service_name="Style")

def call_sdxl_inpainting_service(image_bytes: bytes, mask_bytes: bytes, prompt: str, params: dict = None) -> bytes:
    """Calls the SDXL Inpainting microservice."""
    print("AI_CLIENT: Calling SDXL Inpainting service.")
    url = settings.SDXL_INPAINTING_SERVICE_URL
    files = {
        "base_image": ("base_image.png", image_bytes, "image/png"),
        "mask_image": ("mask_image.png", mask_bytes, "image/png")
    }
    
    data = {"prompt": prompt}
    if params:
        if "guidance_scale" in params: data["guidance_scale"] = params["guidance_scale"]
        if "inference_steps" in params: data["num_inference_steps"] = params["inference_steps"]
        if "strength" in params: data["strength"] = params["strength"]
        if "seed" in params and params["seed"] is not None: data["seed"] = params["seed"]

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, files=files, data=data)
            response.raise_for_status()
            return response.content
    except httpx.HTTPStatusError as e:
        raise AIRequestError(f"SDXL Inpainting service error: {e.response.text}", service_name="SDXL", status_code=e.response.status_code)
    except httpx.RequestError as e:
        raise AIRequestError(f"SDXL Inpainting service connection failed: {str(e)}", service_name="SDXL")
    except Exception as e:
        raise AIRequestError(f"Unexpected error calling SDXL Inpainting service: {str(e)}", service_name="SDXL")

def call_rmbg_service(image_bytes: bytes) -> bytes:
    """Calls the Background Removal microservice."""
    print("AI_CLIENT: Calling Background Removal service.")
    files = {'image': ('image.png', image_bytes, 'image/png')}
    try:
        with httpx.Client(timeout=300.0) as client:
            response = client.post(settings.RMBG_SERVICE_URL, files=files)
            response.raise_for_status()
            return response.content
    except httpx.HTTPStatusError as e:
        raise AIRequestError(f"Background Removal service failed with status {e.response.status_code}: {e.response.text}") from e
    except httpx.RequestError as e:
        raise AIRequestError(f"Could not connect to Background Removal service: {e}") from e

def call_wan_service(prompt: str, negative_prompt: str = "", image_bytes: bytes = None) -> bytes:
    """Calls the Wan Video Generation microservice."""
    print(f"AI_CLIENT: Calling Wan Video service for prompt '{prompt}'")
    url = settings.WAN_SERVICE_URL
    
    data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt
    }
    files = {}
    if image_bytes:
        files["image"] = ("image.png", image_bytes, "image/png")

    try:
        with httpx.Client(timeout=600.0) as client: # Long timeout for video generation
            response = client.post(url, data=data, files=files if files else None)
            response.raise_for_status()
            return response.content
    except httpx.HTTPStatusError as e:
        raise AIRequestError(f"Wan Video service error: {e.response.text}", service_name="Wan", status_code=e.response.status_code)
    except httpx.RequestError as e:
        raise AIRequestError(f"Wan Video service connection failed: {str(e)}", service_name="Wan")
    except Exception as e:
        raise AIRequestError(f"Unexpected error calling Wan Video service: {str(e)}", service_name="Wan")

