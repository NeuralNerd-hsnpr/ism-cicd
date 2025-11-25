import torch
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
from PIL import Image
from io import BytesIO
from typing import Optional, List
from l2cs import Pipeline

# --- Model Management and Configuration ---
model_manager = {}
MODEL_WEIGHTS_PATH = "checkpoints/L2CSNet_gaze360.pkl"

# --- Helper functions adapted from your Gaze class ---
def _classify_yaw(yaw_rad: float, yaw_thr: float = 0.02) -> Optional[str]:
    if yaw_rad > yaw_thr: return "right"
    elif yaw_rad < -yaw_thr: return "left"
    else: return None

def _pil_to_bgr_np(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def _mask_to_bool_np(mask: Image.Image) -> np.ndarray:
    return np.array(mask.convert("L")) > 0

def _select_bbox_index_by_mask(bboxes: List[List[int]], mask_np: np.ndarray) -> Optional[int]:
    best_i, best_count = None, 0
    for i, (x0, y0, x1, y1) in enumerate(bboxes):
        h, w = mask_np.shape
        x0c, y0c = max(0, int(x0)), max(0, int(y0))
        x1c, y1c = min(w, int(x1)), min(h, int(y1))
        region = mask_np[y0c:y1c, x0c:x1c]
        count = int(region.sum())
        if count > best_count:
            best_count = count
            best_i = i
    return best_i

# --- FastAPI Lifespan Manager for Model Loading ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("INFO:     Loading Gaze Estimation model (L2CS)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"INFO:     Using device: {device}")
    try:
        pipeline = Pipeline(
            weights=MODEL_WEIGHTS_PATH,
            arch='ResNet50',
            device=torch.device(device)
        )
        model_manager["gaze_pipeline"] = pipeline
        print("INFO:     Gaze Estimation model loaded successfully.")
    except Exception as e:
        print(f"FATAL:    Failed to load model on startup: {e}")
    yield
    print("INFO:     Cleaning up resources.")
    model_manager.clear()

app = FastAPI(title="Gaze Estimation Service", lifespan=lifespan)

# --- API Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/get_direction_by_mask")
async def get_direction_by_mask(
    base_image: UploadFile = File(..., description="The base image containing the face(s)."),
    face_mask: UploadFile = File(..., description="A mask image where white pixels indicate the target face.")
):
    """
    Estimates the gaze direction ('left', 'right', or 'center') for a face
    specified by a mask.
    """
    pipeline = model_manager.get("gaze_pipeline")
    if not pipeline:
        raise HTTPException(status_code=503, detail="Model is not available or failed to load.")

    try:
        base_image_bytes = await base_image.read()
        face_mask_bytes = await face_mask.read()
        base_image_pil = Image.open(BytesIO(base_image_bytes))
        face_mask_pil = Image.open(BytesIO(face_mask_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file provided.")

    print("INFO:     Running gaze estimation inference...")
    try:
        # This logic is adapted directly from your Gaze.get_direction_by_mask method
        img_np = _pil_to_bgr_np(base_image_pil)
        mask_np = _mask_to_bool_np(face_mask_pil)
        
        results = pipeline.step(img_np)
        
        bboxes = results.bboxes
        idx = _select_bbox_index_by_mask(bboxes, mask_np)
        
        if idx is None:
            direction = "center" # Default to center if no overlapping face is found
        else:
            yaw = float(results.yaw[idx])
            direction = _classify_yaw(yaw) or "center" # Use 'center' if classify_yaw returns None

    except Exception as e:
        print(f"ERROR:    Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred during model inference: {e}")
    
    print(f"INFO:     Inference complete. Detected direction: {direction}")

    return {"direction": direction}