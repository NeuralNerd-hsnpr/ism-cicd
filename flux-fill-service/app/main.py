# app/main.py

from fastapi import FastAPI, HTTPException, Response, File, UploadFile, Form
from contextlib import asynccontextmanager
from PIL import Image
from io import BytesIO
from typing import Optional

# Import your existing ModelManager
from app.models.manager import model_manager

# --- Model Loading on Startup (Lifespan Manager) ---
# This ensures the model is loaded only once.
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("INFO:     Triggering model loading...")
    model_manager.load_models()
    yield
    print("INFO:     Cleaning up resources.")
    # You can add cleanup logic here if needed

app = FastAPI(title="FLUX-Fill Service", lifespan=lifespan)


@app.get("/health")
def health_check():
    return {"status": "ok"}


# --- MODIFIED: The /inpaint endpoint now accepts multipart/form-data ---
@app.post("/inpaint")
async def run_inpaint(
    # --- Image File Inputs ---
    base_image: UploadFile = File(..., description="The base image file."),
    mask_image: UploadFile = File(..., description="The mask image file where the inpainting should occur."),
    
    # --- Required and Optional Form Data ---
    # We use Form() to get data from the same multipart request as the files.
    prompt: str = Form(..., description="The text prompt describing the desired inpainting."),
    seed: Optional[int] = Form(None, description="Seed for reproducibility."),
    guidance_scale: Optional[float] = Form(None, description="Guidance scale for the model."),
    num_inference_steps: Optional[int] = Form(None, description="Number of inference steps.")
):
    """
    Accepts a base image, a mask image, and form data to perform inpainting.
    Returns the resulting image as raw PNG bytes.
    """
    try:
        # 1. Read the uploaded files and convert to PIL Image objects
        print("ENDPOINT: Loading base and mask images from uploaded bytes...")
        base_image_bytes = await base_image.read()
        mask_image_bytes = await mask_image.read()
        
        base_image_pil = Image.open(BytesIO(base_image_bytes))
        mask_image_pil = Image.open(BytesIO(mask_image_bytes))

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file provided. Could not open image data.")
    
    # 2. Prepare optional inference parameters
    # This logic is the same as before, just reading from Form fields.
    inference_params = {}
    if seed is not None:
        inference_params["seed"] = seed
    if guidance_scale is not None:
        inference_params["guidance_scale"] = guidance_scale
    if num_inference_steps is not None:
        inference_params["num_inference_steps"] = num_inference_steps
    
    print(f"ENDPOINT: Running inference with optional params: {inference_params}")

    # 3. Run the inference
    # The run_inference function is unchanged as it already accepts PIL images.
    try:
        output_image_pil = model_manager.run_inference(
            prompt=prompt,
            base_image=base_image_pil,
            mask_image=mask_image_pil,
            **inference_params
        )
    except Exception as e:
        print(f"ERROR:    Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred during model inference: {e}")

    # 4. Convert the output PIL image to bytes
    with BytesIO() as buf:
        output_image_pil.save(buf, format='PNG')
        image_bytes = buf.getvalue()

    # 5. Return the raw image bytes with the correct content type
    return Response(content=image_bytes, media_type="image/png")