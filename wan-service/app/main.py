import io
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
from PIL import Image
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import uvicorn

app = FastAPI(title="Wan Video Generation Service")

# --- Wan Model Class (Adapted from User Code) ---
class Wan:
    def __init__(self):
        print("Initializing Wan Video Pipeline...")
        # Note: Adjust model_id and paths as necessary for the actual environment.
        # Assuming the user has these models available or they will be downloaded.
        # For this implementation, we use the user's provided config.
        self.wan_pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda" if torch.cuda.is_available() else "cpu", # Fallback for safety
            model_configs=[
                ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
                ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
                ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
            ],
        )
        self.wan_pipe.enable_vram_management()
        print("Wan Video Pipeline Initialized.")

    def run_wan_inference(self, image, prompt, negative_prompt):
       # Text-to-video
        if image is None:
            video = self.wan_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=0, tiled=True,
                height=704, width=1248,
                num_frames=121,
            )
        
        # image-to-video
        else:
            image = image.resize((1248, 704)).convert("RGB")
            video = self.wan_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=0, tiled=True,
                height=704, width=1248,
                input_image=image,
                num_frames=121,
            )

        return video

# Initialize model globally
wan_model = None

@app.on_event("startup")
async def startup_event():
    global wan_model
    try:
        wan_model = Wan()
    except Exception as e:
        print(f"Failed to initialize Wan model: {e}")
        # We don't raise here to allow the service to start, but endpoints will fail.

@app.post("/generate_video")
async def generate_video(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    image: Optional[UploadFile] = File(None)
):
    global wan_model
    if wan_model is None:
        raise HTTPException(status_code=503, detail="Wan model is not initialized.")

    try:
        pil_image = None
        if image:
            image_bytes = await image.read()
            pil_image = Image.open(io.BytesIO(image_bytes))

        # Run inference
        # The user's code returns 'video'. We need to know what 'video' is.
        # Usually diffsynth returns a list of frames or a path. 
        # Assuming it returns a list of PIL images or similar based on typical pipelines.
        # However, looking at the user's code, it just says `video = ...`.
        # We will assume for now it returns a list of PIL Images representing frames.
        # We need to save this as a video file (e.g., mp4) to return it.
        
        video_frames = wan_model.run_wan_inference(pil_image, prompt, negative_prompt)
        
        # Save frames to MP4
        # We need a way to convert frames to video. 
        # diffsynth might have a utility, or we use imageio/cv2.
        # Let's assume video_frames is a list of PIL Images.
        
        import tempfile
        import imageio
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
            # Use imageio to write frames to mp4
            # Assuming 24 fps default
            writer = imageio.get_writer(tmp_video.name, fps=24, format='FFMPEG')
            for frame in video_frames:
                # Ensure frame is numpy array
                import numpy as np
                writer.append_data(np.array(frame))
            writer.close()
            
            # Read back the file to return bytes
            with open(tmp_video.name, "rb") as f:
                video_bytes = f.read()
                
        return Response(content=video_bytes, media_type="video/mp4")

    except Exception as e:
        print(f"Error generating video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8007)
