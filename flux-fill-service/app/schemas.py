#Sample payload for this API endpoint:

# {
#   "base_image_url": "http://...",
#   "mask_url": "http://...",
#   "prompt": "a majestic cat on a throne",
#   "seed": 12345,
#   "num_inference_steps": 30
# }


from pydantic import BaseModel
from typing import Optional

class InpaintRequest(BaseModel):
    """
    Updated schema for an inpainting request.
    Includes optional parameters to control the inference process.
    """
    # Required fields remain the same
    base_image_url: str
    mask_url: str
    prompt: str

    # --- NEW: Optional inference parameters ---
    # By setting the default to None, these fields become optional in the API request.
    seed: Optional[int] = None
    guidance_scale: Optional[float] = None
    num_inference_steps: Optional[int] = None

