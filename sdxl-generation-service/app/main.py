import torch
from fastapi import FastAPI, Form, HTTPException, Response
from contextlib import asynccontextmanager
from PIL import Image
from io import BytesIO
from typing import Optional
from diffusers import DiffusionPipeline

# A simple dictionary to hold our loaded models
model_manager = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic: Load both SDXL models ---
    print("INFO:     Loading SDXL Base and Refiner models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"INFO:     Using device: {device}")

    try:
        # 1. Load the Base SDXL model
        print("INFO:     Loading SDXL Base pipeline...")
        base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch_dtype,
            variant="fp16",
            use_safetensors=True
        )
        base.to(device)
        model_manager["base"] = base

        # 2. Load the Refiner, sharing components for VRAM efficiency
        print("INFO:     Loading SDXL Refiner pipeline...")
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant="fp16",
        )
        refiner.to(device)
        model_manager["refiner"] = refiner
        model_manager["device"] = device
        
        print("INFO:     SDXL Base and Refiner models loaded successfully.")
    except Exception as e:
        print(f"FATAL:    Failed to load models on startup: {e}")
    
    yield
    # --- Shutdown Logic ---
    print("INFO:     Cleaning up resources.")
    model_manager.clear()

app = FastAPI(title="SDXL Image Generation Service", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/generate")
async def run_sdxl_generation(
    prompt: str = Form(..., description="The text prompt to generate an image from."),
    # Add optional parameters for finer control
    negative_prompt: Optional[str] = Form(None, description="The negative prompt to guide the model away from."),
    num_inference_steps: int = Form(40, ge=20, le=100),
    high_noise_frac: float = Form(0.8, ge=0.5, le=1.0, description="Fraction of steps to run on the base model."),
    seed: Optional[int] = Form(None)
):
    """
    Generates a high-quality image from a text prompt using the SDXL
    Base + Refiner pipeline and returns the image as raw PNG bytes.
    """
    base = model_manager.get("base")
    refiner = model_manager.get("refiner")
    device = model_manager.get("device")
    if not all([base, refiner, device]):
        raise HTTPException(status_code=503, detail="Models are not available or failed to load.")

    print("INFO:     Running SDXL generation inference...")
    try:
        # Use a generator for reproducibility if a seed is provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
            
        # --- This logic is adapted directly from your provided code ---
        
        # 1. Run the base model to generate latent representations
        latents = base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
            generator=generator,
        ).images

        # 2. Run the refiner model on the latents to produce the final image
        output_image_pil = refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            denoising_start=high_noise_frac,
            image=latents,
            generator=generator,
        ).images[0]

    except Exception as e:
        print(f"ERROR:    Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred during model inference: {e}")
    
    print("INFO:     Inference complete.")

    with BytesIO() as buf:
        output_image_pil.save(buf, format='PNG')
        output_image_bytes = buf.getvalue()

    return Response(content=output_image_bytes, media_type="image/png")