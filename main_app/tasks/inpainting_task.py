import httpx
import uuid
from tasks.celery_app import celery_app
from services import azure_service, api_service 
from core.config import settings
from services import image_utils
import time
from io import BytesIO

class InpaintingOrchestrationError(Exception):
    """Custom exception for failures during the inpainting orchestration."""
    pass

@celery_app.task(bind=True)
def process_complex_inpainting_request(self, request_data: dict):
    """
    Orchestrates the entire inpainting pipeline for a complex request.

    This version is fully updated to call the refactored Flux API, which
    accepts image bytes directly via a multipart/form-data request,
    while preserving all original business logic.
    """
    full_start_time = time.perf_counter()
    task_id = self.request.id

    try:
        # --- 1. Initialization and Pre-processing ---
        parent_id = request_data.get("parent_id")
        if not parent_id:
            print("WARNING: parent_id not found in request. Generating a random UUID.")
            parent_id = str(uuid.uuid4())

        images = request_data.get("images", [])
        output_verbosity = request_data.get("output", "default").lower()

        final_output_data = {
            "parentID": parent_id,
            "elapsedSeconds": 0,
            "steps": [],
            "results": [],
        }

        upload_input_start_time = time.perf_counter()
        input_filename = f"{task_id}_input.json"
        azure_service.upload_json(data=request_data, file_name=input_filename, is_input=True)
        upload_input_time = time.perf_counter() - upload_input_start_time

        # --- 2. Core Image Processing Loop ---
        processing_start_time = time.perf_counter()
        all_image_results = []

        for image in images:
            image_id = image.get("image_id")
            base_image_url = image.get("base_url")

            if not base_image_url:
                print(f"ERROR: Skipping image '{image_id}' due to missing 'base_url'.")
                continue

            for operation in image.get("operations", []):
                phrase = operation.get("phrase")
                mask_url = operation.get("mask_url")

                if not all([phrase, mask_url]):
                    print(f"ERROR: Skipping operation for image '{image_id}' due to missing 'phrase' or 'mask_url'.")
                    continue

                # The `model` key is no longer relevant for this simple pipeline, but we keep the structure.
                model = operation.get("model", "flux").lower()
                
                if model == "sdxl":
                    try:
                        # --- SDXL Logic ---
                        network_start_time = time.perf_counter()
                        print(f"Downloading images for task {task_id} (SDXL)...")
                        base_image_pil = image_utils.load_image_from_url(base_image_url)
                        mask_pil = image_utils.load_image_from_url(mask_url)
                        network_io_time = time.perf_counter() - network_start_time

                        base_image_bytes = BytesIO(); base_image_pil.save(base_image_bytes, format='PNG')
                        mask_bytes = BytesIO(); mask_pil.save(mask_bytes, format='PNG')

                        sdxl_params = {
                            "inference_steps": operation.get("num_inference_steps"),
                            "guidance_scale": operation.get("guidance_scale"),
                            "strength": operation.get("strength", 0.8),
                            "seed": operation.get("seed")
                        }

                        inference_start_time = time.perf_counter()
                        output_image_bytes = api_service.call_sdxl_inpainting_service(
                            image_bytes=base_image_bytes.getvalue(),
                            mask_bytes=mask_bytes.getvalue(),
                            prompt=phrase,
                            params=sdxl_params
                        )
                        inpaint_time = time.perf_counter() - inference_start_time

                        # Save result
                        file_name = f"{task_id}_{image_id}_{phrase}_sdxl_inpainted.png"
                        image_urls = azure_service.save_image_locally_and_get_urls(
                            image_bytes=output_image_bytes, file_name=file_name
                        )

                        operation_result = {
                            "imageID": image_id, "success": True, "error": None,
                            "imageTemporary": image_urls["public_url"],
                            "imagePermanent": image_urls["permanent_url"],
                            "steps": [
                                 {"step": 1, "action": "in-painting", "aiModels": "sdxl", "methods": None, "version": 1.0, "elapsedSeconds": round(inpaint_time, 2)},
                                 {"step": 2, "action": "total_network_runtime", "methods": "requests", "elapsedSeconds": round(network_io_time, 2)}
                            ]
                        }
                        all_image_results.append(operation_result)

                    except Exception as e:
                        print(f"CRITICAL ERROR: SDXL Inpainting failed: {e}")
                        user_error_message = f"The AI inpainting service (SDXL) failed: {str(e)}"
                        all_image_results.append({
                            "imageID": image_id, "success": False, "error": user_error_message,
                            "imageTemporary": None, "imagePermanent": None, "steps": []
                        })
                        raise InpaintingOrchestrationError(user_error_message) from e

                else: # Default to Flux
                    try:
                        # --- Flux Logic ---
                        network_start_time = time.perf_counter()
                        print(f"Downloading images for task {task_id} (Flux)...")
                        base_image_pil = image_utils.load_image_from_url(base_image_url)
                        mask_pil = image_utils.load_image_from_url(mask_url)
                        network_io_time = time.perf_counter() - network_start_time

                        base_image_bytes = BytesIO(); base_image_pil.save(base_image_bytes, format='PNG')
                        mask_bytes = BytesIO(); mask_pil.save(mask_bytes, format='PNG')
                        
                        flux_params = {
                            "seed": operation.get("seed"),
                            "guidance_scale": operation.get("guidance_scale"),
                            "num_inference_steps": operation.get("num_inference_steps")
                        }

                        inference_start_time = time.perf_counter()
                        output_image_bytes = api_service.call_flux_service(
                            image_bytes=base_image_bytes.getvalue(),
                            mask_bytes=mask_bytes.getvalue(),
                            prompt=phrase,
                            params=flux_params
                        )
                        inpaint_time = time.perf_counter() - inference_start_time

                        # Save result
                        file_name = f"{task_id}_{image_id}_{phrase}_inpainted.png"
                        image_urls = azure_service.save_image_locally_and_get_urls(
                            image_bytes=output_image_bytes, file_name=file_name
                        )

                        operation_result = {
                            "imageID": image_id, "success": True, "error": None,
                            "imageTemporary": image_urls["public_url"],
                            "imagePermanent": image_urls["permanent_url"],
                            "steps": [
                                 {"step": 1, "action": "in-painting", "aiModels": "flux_fill_dev", "methods": None, "version": 1.0, "elapsedSeconds": round(inpaint_time, 2)},
                                 {"step": 2, "action": "total_network_runtime", "methods": "requests", "elapsedSeconds": round(network_io_time, 2)}
                            ]
                        }
                        all_image_results.append(operation_result)

                    except Exception as e:
                        print(f"CRITICAL ERROR: Flux Inpainting failed: {e}")
                        user_error_message = f"The AI inpainting service failed: {str(e)}"
                        all_image_results.append({
                            "imageID": image_id, "success": False, "error": user_error_message,
                            "imageTemporary": None, "imagePermanent": None, "steps": []
                        })
                        raise InpaintingOrchestrationError(user_error_message) from e

        # --- This section is identical to your latest version ---
        final_output_data["results"] = all_image_results
        total_process_time = time.perf_counter() - processing_start_time

        # 3. Finalization (only reached on success)
        total_elapsed_time = time.perf_counter() - full_start_time
        final_output_data["elapsedSeconds"] = round(total_elapsed_time, 2)
        final_output_data["steps"] = [
            {"step": 1, "action": "upload request to blob storage", "elapsedSeconds": round(upload_input_time, 2)},
            {"step": 2, "action": "image processing", "elapsedSeconds": round(total_process_time, 2)}
        ]

        upload_output_start_time = time.perf_counter()
        output_filename = f"{task_id}_output.json"
        azure_service.upload_json(data=final_output_data, file_name=output_filename, is_input=False)
        upload_output_time = time.perf_counter() - upload_output_start_time
        final_output_data["steps"].append({"step": 3, "action": "upload response to blob storage", "elapsedSeconds": round(upload_output_time, 2)})

        # 4. Prune Output and Return Success
        if output_verbosity == "default":
            final_output_data.pop("steps", None)
            pruned_results = []
            keys_to_include = {"imageID", "imageTemporary", "imagePermanent"}
            for result in final_output_data["results"]:
                if result.get("success"):
                    pruned_results.append({key: result[key] for key in keys_to_include if key in result})
                else:
                    pruned_results.append(result)
            final_output_data["results"] = pruned_results
        
        return {"success": True, "output": final_output_data}

    except Exception as e:
        print(f"FATAL PIPELINE ERROR: Inpainting task {task_id} failed with exception: {e}")
        error_report = {
            "parentID": request_data.get("parent_id"),
            "taskID": task_id,
            "status": "FAILURE",
            "error": f"A fatal error occurred in the pipeline: {str(e)}"
        }
        azure_service.upload_json(data=error_report, file_name=f"{task_id}_error.json", is_input=False)
        raise