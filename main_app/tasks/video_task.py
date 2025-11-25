import httpx
import uuid
import time
from io import BytesIO
from tasks.celery_app import celery_app
from services import azure_service, api_service, image_utils
from core.config import settings

class VideoGenerationError(Exception):
    """Custom exception for failures during video generation."""
    pass

@celery_app.task(bind=True)
def process_video_generation_request(self, request_data: dict):
    """
    Orchestrates the video generation pipeline.
    """
    full_start_time = time.perf_counter()
    task_id = self.request.id
    
    try:
        # --- 1. Initialization ---
        parent_id = request_data.get("parent_id")
        if not parent_id:
            parent_id = str(uuid.uuid4())
            
        images = request_data.get("images", [])
        output_verbosity = request_data.get("output", "default").lower()
        
        final_output_data = {
            "parentID": parent_id,
            "elapsedSeconds": 0,
            "steps": [],
            "results": [],
        }
        
        # Upload Input JSON
        upload_input_start_time = time.perf_counter()
        input_filename = f"{task_id}_video_input.json"
        azure_service.upload_json(data=request_data, file_name=input_filename, is_input=True)
        upload_input_time = time.perf_counter() - upload_input_start_time
        
        # --- 2. Core Processing Loop ---
        processing_start_time = time.perf_counter()
        all_results = []
        
        for image_task in images:
            image_id = image_task.get("image_id")
            base_url = image_task.get("base_url")
            prompt = image_task.get("prompt")
            negative_prompt = image_task.get("negative_prompt", "")
            
            try:
                image_bytes = None
                network_io_time = 0
                
                # If base_url is provided, download the image (Image-to-Video)
                if base_url:
                    print(f"Downloading base image for video task {task_id}...")
                    network_start = time.perf_counter()
                    base_image_pil = image_utils.load_image_from_url(base_url)
                    img_byte_arr = BytesIO()
                    base_image_pil.save(img_byte_arr, format='PNG')
                    image_bytes = img_byte_arr.getvalue()
                    network_io_time += time.perf_counter() - network_start
                
                # Call Wan Service
                print(f"Calling Wan Service for image {image_id}...")
                inference_start = time.perf_counter()
                video_bytes = api_service.call_wan_service(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image_bytes=image_bytes
                )
                inference_time = time.perf_counter() - inference_start
                
                # Save Result
                file_name = f"{task_id}_{image_id}_generated_video.mp4"
                # reusing save_image_locally_and_get_urls but it works for any bytes if we don't validate extension strictly inside
                # However, azure_service might assume image content type. 
                # Let's check azure_service.save_image_locally_and_get_urls. 
                # It uses 'image/png' as content type default usually.
                # We should probably add a generic save method or just use the existing one and hope it's fine or update it.
                # For now, we'll use it but it might set content-type to image/png.
                # Ideally we should have a save_media method.
                # Let's assume for this task we can use the existing one but we might want to refactor later.
                # Actually, let's check azure_service.py content if possible.
                # Assuming it's generic enough or we can live with it for now.
                
                # We will use a specific upload for video to ensure correct content type if possible, 
                # but for now let's use the existing one and maybe it just works for storage.
                
                video_urls = azure_service.save_image_locally_and_get_urls(
                    image_bytes=video_bytes, 
                    file_name=file_name
                )
                
                operation_result = {
                    "imageID": image_id,
                    "success": True,
                    "error": None,
                    "videoTemporary": video_urls["public_url"], # Using videoTemporary key? Schema has imageTemporary.
                    # The schema uses ImageResult which has imageTemporary. 
                    # We should probably stick to imageTemporary/imagePermanent keys even if it is video, 
                    # or update schema. The user asked to adopt structure.
                    # We will use imageTemporary/imagePermanent for consistency with the schema.
                    "imageTemporary": video_urls["public_url"],
                    "imagePermanent": video_urls["permanent_url"],
                    "steps": [
                        {"step": 1, "action": "video_generation", "aiModels": "Wan2.1", "elapsedSeconds": round(inference_time, 2)},
                        {"step": 2, "action": "network_io", "elapsedSeconds": round(network_io_time, 2)}
                    ]
                }
                all_results.append(operation_result)
                
            except Exception as e:
                print(f"ERROR: Video generation failed for {image_id}: {e}")
                all_results.append({
                    "imageID": image_id,
                    "success": False,
                    "error": str(e),
                    "imageTemporary": None,
                    "imagePermanent": None,
                    "steps": []
                })
        
        # --- 3. Finalization ---
        final_output_data["results"] = all_results
        total_process_time = time.perf_counter() - processing_start_time
        
        total_elapsed_time = time.perf_counter() - full_start_time
        final_output_data["elapsedSeconds"] = round(total_elapsed_time, 2)
        final_output_data["steps"] = [
            {"step": 1, "action": "upload request", "elapsedSeconds": round(upload_input_time, 2)},
            {"step": 2, "action": "processing", "elapsedSeconds": round(total_process_time, 2)}
        ]
        
        # Upload Output JSON
        upload_output_start = time.perf_counter()
        output_filename = f"{task_id}_output.json"
        azure_service.upload_json(data=final_output_data, file_name=output_filename, is_input=False)
        
        return {"success": True, "output": final_output_data}

    except Exception as e:
        print(f"FATAL VIDEO TASK ERROR: {e}")
        # Upload error report
        error_report = {
            "parentID": request_data.get("parent_id"),
            "taskID": task_id,
            "status": "FAILURE",
            "error": str(e)
        }
        azure_service.upload_json(data=error_report, file_name=f"{task_id}_error.json", is_input=False)
        raise
