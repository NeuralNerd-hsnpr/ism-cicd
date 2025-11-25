import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from contextlib import asynccontextmanager
from PIL import Image
from io import BytesIO
from typing import Optional
from enum import Enum

# Import all necessary components from the diffusers library
from diffusers import (
    ControlNetModel,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionControlNetPipeline,
    EulerAncestralDiscreteScheduler,
    UniPCMultistepScheduler
)

# A simple dictionary to hold our loaded models
model_manager = {}

# An Enum for clear model selection in the API
class StyleModelChoice(str, Enum):
    PIX2PIX = "pix2pix"
    CONTROLNET = "controlnet"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic: Load both models into memory ---
    print("INFO:     Loading Stable Diffusion models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    try:
        # 1. Load the InstructPix2Pix model
        print("INFO:     Loading InstructPix2Pix pipeline...")
        pix2pix_id = "timbrooks/instruct-pix2pix"
        pix_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            pix2pix_id, torch_dtype=torch_dtype, safety_checker=None
        )
        pix_pipe.to(device)
        pix_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pix_pipe.scheduler.config)
        model_manager[StyleModelChoice.PIX2PIX] = pix_pipe
        print("INFO:     InstructPix2Pix loaded successfully.")

        # 2. Load the ControlNet model for instruction-following
        print("INFO:     Loading ControlNet InstructPix2Pix pipeline...")
        controlnet_model = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11e_sd15_ip2p", torch_dtype=torch_dtype
        )
        control_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet_model, torch_dtype=torch_dtype
        )
        control_pipe.to(device)
        control_pipe.scheduler = UniPCMultistepScheduler.from_config(control_pipe.scheduler.config)
        model_manager[StyleModelChoice.CONTROLNET] = control_pipe
        print("INFO:     ControlNet loaded successfully.")
        
    except Exception as e:
        print(f"FATAL:    Failed to load models on startup: {e}")
        # In a real production system, you might want the app to fail to start
        # if the models can't be loaded.
    
    yield
    # --- Shutdown Logic ---
    print("INFO:     Cleaning up model resources.")
    model_manager.clear()

app = FastAPI(title="Stable Diffusion Styling Service", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/style")
async def run_style_transfer(
    image: UploadFile = File(..., description="The base image file to be stylized."),
    prompt: str = Form(..., description="The instruction prompt describing the style change."),
    model: StyleModelChoice = Form(..., description="The model to use for styling."),
    num_inference_steps: int = Form(50, ge=10, le=150, description="Number of diffusion steps."),
    image_guidance_scale: float = Form(1.5, ge=1.0, le=5.0, description="Image guidance scale (only for 'pix2pix' model).")
):
    """
    Accepts an image and an instruction prompt, then applies the style
    using the selected Stable Diffusion model.
    """
    pipeline = model_manager.get(model)
    if not pipeline:
        raise HTTPException(status_code=503, detail=f"Model '{model.value}' is not available or failed to load.")

    try:
        image_bytes = await image.read()
        image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file provided.")

    print(f"INFO:     Running inference with model '{model.value}'...")
    try:
        # Each pipeline has a slightly different call signature
        if model == StyleModelChoice.PIX2PIX:
            output_image_pil = pipeline(
                prompt,
                image=image_pil,
                num_inference_steps=num_inference_steps,
                image_guidance_scale=image_guidance_scale
            ).images[0]
        elif model == StyleModelChoice.CONTROLNET:
            # ControlNet doesn't use image_guidance_scale but can use a generator for reproducibility
            generator = torch.Generator(device="cuda").manual_seed(42) # Example seed
            output_image_pil = pipeline(
                prompt,
                image=image_pil,
                num_inference_steps=num_inference_steps,
                generator=generator
            ).images[0]

    except Exception as e:
        print(f"ERROR:    Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred during model inference: {e}")
    
    print("INFO:     Inference complete.")

    with BytesIO() as buf:
        output_image_pil.save(buf, format='PNG')
        output_image_bytes = buf.getvalue()

    return Response(content=output_image_bytes, media_type="image/png")