import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from contextlib import asynccontextmanager
from PIL import Image
from io import BytesIO
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# A simple dictionary to hold our model and device context
model_manager = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic: Load the background removal model ---
    print("INFO:     Loading Background Removal model (briaai/RMBG-2.0)...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"INFO:     Using device: {device}")
    
    try:
        model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-2.0", trust_remote_code=True
        )
        torch.set_float32_matmul_precision("high")
        model.to(device)
        model.eval()
        
        model_manager["bg_model"] = model
        model_manager["device"] = device
        print("INFO:     Background Removal model loaded successfully.")
    except Exception as e:
        print(f"FATAL:    Failed to load model on startup: {e}")
    
    yield
    # --- Shutdown Logic ---
    print("INFO:     Cleaning up resources.")
    model_manager.clear()

app = FastAPI(title="Background Removal Service", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/remove_background")
async def run_background_removal(
    image: UploadFile = File(..., description="The image file from which to remove the background.")
):
    """
    Accepts an image, removes the background, and returns the resulting
    image with a transparent background as a PNG.
    """
    model = model_manager.get("bg_model")
    device = model_manager.get("device")
    if not all([model, device]):
        raise HTTPException(status_code=503, detail="Model is not available or failed to load.")

    try:
        image_bytes = await image.read()
        image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file provided.")

    print("INFO:     Running background removal inference...")
    try:
        # --- This logic is adapted directly from your provided code ---
        original_size = image_pil.size
        
        # 1. Prepare the image for the model
        image_size = (1024, 1024) # The model's expected input size
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        transformed_image = transform(image_pil).unsqueeze(0).to(device)

        # 2. Run prediction to get the mask
        with torch.no_grad():
            preds = model(transformed_image)[-1].sigmoid().cpu()
        
        # 3. Post-process the mask
        pred_pil = transforms.ToPILImage()(preds[0].squeeze())
        mask = pred_pil.resize(original_size, Image.LANCZOS) # Resize mask back to original image size
        
        # 4. Apply the mask as an alpha channel to the original image
        output_image_pil = image_pil.copy()
        output_image_pil.putalpha(mask)

    except Exception as e:
        print(f"ERROR:    Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred during model inference: {e}")
    
    print("INFO:     Inference complete.")

    # 5. Convert the final RGBA image to bytes and return
    with BytesIO() as buf:
        # Save as PNG to preserve the alpha (transparency) channel
        output_image_pil.save(buf, format='PNG')
        output_image_bytes = buf.getvalue()

    return Response(content=output_image_bytes, media_type="image/png")