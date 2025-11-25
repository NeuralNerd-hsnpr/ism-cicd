from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from contextlib import asynccontextmanager
from PIL import Image
from io import BytesIO
from simple_lama_inpainting import SimpleLama

# A simple dictionary to hold our model so it's globally accessible
model_manager = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    print("INFO:     Loading SimpleLama model...")
    # This is the slow part that runs only once.
    model_manager["lama_model"] = SimpleLama()
    print("INFO:     SimpleLama model loaded successfully.")
    yield
    # --- Shutdown Logic ---
    print("INFO:     Cleaning up resources.")
    model_manager.clear()

app = FastAPI(title="Simple Lama Removal Service", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/remove")
async def run_object_removal(
    image: UploadFile = File(..., description="The base image file."),
    mask: UploadFile = File(..., description="The mask image file where white pixels indicate the area to remove.")
):
    """
    Accepts a base image and a mask image, and returns the image with the
    masked object removed.
    """
    try:
        # 1. Read the uploaded files into memory as bytes
        image_bytes = await image.read()
        mask_bytes = await mask.read()

        # 2. Convert the bytes into PIL Image objects
        image_pil = Image.open(BytesIO(image_bytes))
        mask_pil = Image.open(BytesIO(mask_bytes)).convert("L") # Ensure mask is grayscale

    except Exception:
        # This catches errors if the uploaded files are not valid images
        raise HTTPException(status_code=400, detail="Invalid image file provided. Could not open image.")

    # 3. Run the inference using the pre-loaded model
    print("INFO:     Running Lama inference...")
    lama_model = model_manager.get("lama_model")
    if not lama_model:
        raise HTTPException(status_code=503, detail="Model is not available or failed to load.")

    try:
        output_image_pil = lama_model(image_pil, mask_pil)
    except Exception as e:
        print(f"ERROR:    Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred during model inference: {e}")
    
    print("INFO:     Inference complete.")

    # 4. Convert the output PIL image back to bytes to send in the response
    with BytesIO() as buf:
        output_image_pil.save(buf, format='PNG')
        output_image_bytes = buf.getvalue()

    # 5. Return the raw image bytes with the correct content type
    return Response(content=output_image_bytes, media_type="image/png")