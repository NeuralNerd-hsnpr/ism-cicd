import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from contextlib import asynccontextmanager
from PIL import Image
from io import BytesIO
from typing import Optional
from diffusers import AutoPipelineForInpainting

# A simple dictionary to hold our loaded model
model_manager = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic: Load the Stable Diffusion Inpainting model ---
    print("INFO:     Loading SDXL Inpainting model (diffusers/stable-diffusion-xl-1.0-inpainting-0.1)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"INFO:     Using device: {device}")

    try:
        pipeline = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch_dtype,
            variant="fp16",
        )
        # enable_model_cpu_offload is great for saving VRAM on cards with less memory
        pipeline.enable_model_cpu_offload()
        
        model_manager["inpainting_pipeline"] = pipeline
        model_manager["device"] = device
        print("INFO:     SDXL Inpainting model loaded successfully.")
    except Exception as e:
        print(f"FATAL:    Failed to load model on startup: {e}")
    
    yield
    # --- Shutdown Logic ---
    print("INFO:     Cleaning up resources.")
    model_manager.clear()

app = FastAPI(title="Stable Diffusion Inpainting Service", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/inpaint")
async def run_sd_inpainting(
    base_image: UploadFile = File(..., description="The base image file."),
    mask_image: UploadFile = File(..., description="The mask image where inpainting should occur."),
    prompt: str = Form(..., description="The text prompt describing the desired inpainting."),
    # Add all other inference parameters as optional Form fields with defaults
    guidance_scale: float = Form(7.5, ge=1.0, le=15.0),
    num_inference_steps: int = Form(30, ge=10, le=100),
    strength: float = Form(0.8, ge=0.1, le=1.0),
    seed: Optional[int] = Form(None)
):
    """
    Accepts a base image, a mask, and a prompt to perform inpainting using
    Stable Diffusion XL and returns the resulting image.
    """
    pipeline = model_manager.get("inpainting_pipeline")
    device = model_manager.get("device")
    if not all([pipeline, device]):
        raise HTTPException(status_code=503, detail="Model is not available or failed to load.")

    try:
        image_bytes = await base_image.read()
        mask_bytes = await mask_image.read()
        image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
        mask_pil = Image.open(BytesIO(mask_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file provided.")

    print("INFO:     Running SDXL Inpainting inference...")
    try:
        # Use a generator for reproducibility if a seed is provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        output_image_pil = pipeline(
            prompt=prompt,
            image=image_pil,
            mask_image=mask_pil,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            strength=strength,
        ).images[0]

    except Exception as e:
        print(f"ERROR:    Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred during model inference: {e}")
    
    print("INFO:     Inference complete.")

    with BytesIO() as buf:
        output_image_pil.save(buf, format='PNG')
        output_image_bytes = buf.getvalue()

    return Response(content=output_image_bytes, media_type="image/png")