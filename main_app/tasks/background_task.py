import httpx
import uuid
import time
from io import BytesIO
from tasks.celery_app import celery_app
from services import azure_service, api_service, image_utils
from core.config import settings
from PIL import Image

@celery_app.task(bind=True)
def process_background_removal_request(self, request_data: dict):
    """
    Orchestrates the background removal pipeline.
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
            image_id = image.get("imageID") or image.get("image_id")
            base_image_url = image.get("url") or image.get("base_url")

            if not base_image_url:
                print(f"ERROR: Skipping image '{image_id}' due to missing 'url'.")
                continue

            try:
                # 1. Download image
                network_start_time = time.perf_counter()
                print(f"Downloading image {image_id} for task {task_id}...")
                base_image_pil = image_utils.load_image_from_url(base_image_url)
                network_io_time = time.perf_counter() - network_start_time

                # 2. Convert to bytes
                base_image_bytes = BytesIO()
                base_image_pil.save(base_image_bytes, format='PNG')
                
                # 3. Call RMBG Service
                inference_start_time = time.perf_counter()
                output_image_bytes = api_service.call_rmbg_service(base_image_bytes.getvalue())
                rmbg_time = time.perf_counter() - inference_start_time
                
                # 4. Save result
                file_name = f"{task_id}_{image_id}_background_removed.png"
                image_urls = azure_service.save_image_locally_and_get_urls(
                    image_bytes=output_image_bytes, file_name=file_name
                )

                # Build success result
                operation_result = {
                    "imageID": image_id,
                    "success": True,
                    "error": None,
                    "bgRemovedTemporary": image_urls["public_url"],
                    "bgRemovedPermanent": image_urls["permanent_url"],
                    "steps": [
                        {"step": 1, "action": "background_remove", "aiModels": "RMBG-2.0", "methods": None, "version": 2.0, "elapsedSeconds": round(rmbg_time, 2)},
                    ]
                }
                all_image_results.append(operation_result)

            except Exception as e:
                print(f"ERROR: Background removal failed for image {image_id}: {e}")
                user_error_message = f"Failed to remove background: {str(e)}"
                all_image_results.append({
                    "imageID": image_id, "success": False, "error": user_error_message,
                    "bgRemovedTemporary": None, "bgRemovedPermanent": None, "steps": []
                })

        final_output_data["results"] = all_image_results
        total_process_time = time.perf_counter() - processing_start_time

        # 3. Finalization
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

        # 4. Prune Output
        if output_verbosity == "default":
            final_output_data.pop("steps", None)
            pruned_results = []
            keys_to_include = {"imageID", "bgRemovedTemporary", "bgRemovedPermanent"}
            for result in final_output_data["results"]:
                if result.get("success"):
                    pruned_results.append({key: result[key] for key in keys_to_include if key in result})
                else:
                    pruned_results.append(result)
            final_output_data["results"] = pruned_results
        
        return {"success": True, "output": final_output_data}

    except Exception as e:
        print(f"FATAL PIPELINE ERROR: Background removal task {task_id} failed with exception: {e}")
        error_report = {
            "parentID": request_data.get("parent_id"),
            "taskID": task_id,
            "status": "FAILURE",
            "error": f"A fatal error occurred in the pipeline: {str(e)}"
        }
        azure_service.upload_json(data=error_report, file_name=f"{task_id}_error.json", is_input=False)
        raise

@celery_app.task(bind=True)
def process_background_insertion_request(self, request_data: dict):
    """
    Orchestrates the background insertion pipeline.
    Assumes the input 'url' points to a transparent foreground image.
    """
    full_start_time = time.perf_counter()
    task_id = self.request.id

    try:
        # --- 1. Initialization ---
        parent_id = request_data.get("parent_id") or str(uuid.uuid4())
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

        # --- 2. Processing Loop ---
        processing_start_time = time.perf_counter()
        all_image_results = []

        for image in images:
            image_id = image.get("imageID") or image.get("image_id")
            base_url = image.get("url") or image.get("base_url")
            insertion_url = image.get("insertionUrl") or image.get("insertion_url")

            if not all([image_id, base_url, insertion_url]):
                print(f"ERROR: Skipping image '{image_id}' due to missing fields.")
                continue

            try:
                # 1. Download images
                network_start_time = time.perf_counter()
                print(f"Downloading images for {image_id}...")
                foreground_pil = image_utils.load_image_from_url(base_url)
                background_pil = image_utils.load_image_from_url(insertion_url)
                network_io_time = time.perf_counter() - network_start_time

                # 2. Composite
                comp_start_time = time.perf_counter()
                output_image_pil = image_utils.composite_images(foreground_pil, background_pil)
                comp_time = time.perf_counter() - comp_start_time

                # 3. Save result
                output_image_bytes = BytesIO()
                output_image_pil.save(output_image_bytes, format='PNG')
                
                file_name = f"{task_id}_{image_id}_background_inserted.png"
                image_urls = azure_service.save_image_locally_and_get_urls(
                    image_bytes=output_image_bytes.getvalue(), file_name=file_name
                )

                # Build result
                operation_result = {
                    "imageID": image_id,
                    "success": True,
                    "error": None,
                    "bgInsertedTemporary": image_urls["public_url"],
                    "bgInsertedPermanent": image_urls["permanent_url"],
                    "steps": [
                        {"step": 1, "action": "background_insertion", "aiModels": None, "methods": "alpha_composite", "version": None, "elapsedSeconds": round(comp_time, 2)},
                    ]
                }
                all_image_results.append(operation_result)

            except Exception as e:
                print(f"ERROR: Background insertion failed for image {image_id}: {e}")
                all_image_results.append({
                    "imageID": image_id, "success": False, "error": str(e),
                    "bgInsertedTemporary": None, "bgInsertedPermanent": None, "steps": []
                })

        final_output_data["results"] = all_image_results
        total_process_time = time.perf_counter() - processing_start_time

        # 3. Finalization
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

        # 4. Prune Output
        if output_verbosity == "default":
            final_output_data.pop("steps", None)
            pruned_results = []
            keys_to_include = {"imageID", "bgInsertedTemporary", "bgInsertedPermanent"}
            for result in final_output_data["results"]:
                if result.get("success"):
                    pruned_results.append({key: result[key] for key in keys_to_include if key in result})
                else:
                    pruned_results.append(result)
            final_output_data["results"] = pruned_results
        
        return {"success": True, "output": final_output_data}

    except Exception as e:
        print(f"FATAL PIPELINE ERROR: Background insertion task {task_id} failed: {e}")
        error_report = {
            "parentID": request_data.get("parent_id"),
            "taskID": task_id,
            "status": "FAILURE",
            "error": str(e)
        }
        azure_service.upload_json(data=error_report, file_name=f"{task_id}_error.json", is_input=False)
        raise

@celery_app.task(bind=True)
def process_background_replacement_request(self, request_data: dict):
    """
    Orchestrates the background replacement pipeline.
    1. Removes background from original image.
    2. Inserts new background.
    """
    full_start_time = time.perf_counter()
    task_id = self.request.id

    try:
        # --- 1. Initialization ---
        parent_id = request_data.get("parent_id") or str(uuid.uuid4())
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

        # --- 2. Processing Loop ---
        processing_start_time = time.perf_counter()
        all_image_results = []

        for image in images:
            image_id = image.get("imageID") or image.get("image_id")
            base_url = image.get("url") or image.get("base_url")
            insertion_url = image.get("insertionUrl") or image.get("insertion_url")

            if not all([image_id, base_url, insertion_url]):
                print(f"ERROR: Skipping image '{image_id}' due to missing fields.")
                continue

            try:
                # 1. Download Original Image
                network_start_time = time.perf_counter()
                print(f"Downloading original image {image_id}...")
                original_pil = image_utils.load_image_from_url(base_url)
                
                # 2. Download Background Image
                print(f"Downloading background image for {image_id}...")
                background_pil = image_utils.load_image_from_url(insertion_url)
                network_io_time = time.perf_counter() - network_start_time

                # 3. Remove Background
                rmbg_start_time = time.perf_counter()
                original_bytes = BytesIO()
                original_pil.save(original_bytes, format='PNG')
                
                print(f"Removing background for {image_id}...")
                foreground_bytes_data = api_service.call_rmbg_service(original_bytes.getvalue())
                foreground_pil = Image.open(BytesIO(foreground_bytes_data)).convert("RGBA")
                rmbg_time = time.perf_counter() - rmbg_start_time

                # 4. Composite
                comp_start_time = time.perf_counter()
                print(f"Compositing images for {image_id}...")
                output_image_pil = image_utils.composite_images(foreground_pil, background_pil)
                comp_time = time.perf_counter() - comp_start_time

                # 5. Save Result
                output_image_bytes = BytesIO()
                output_image_pil.save(output_image_bytes, format='PNG')
                
                file_name = f"{task_id}_{image_id}_background_replaced.png"
                image_urls = azure_service.save_image_locally_and_get_urls(
                    image_bytes=output_image_bytes.getvalue(), file_name=file_name
                )

                # Build result
                operation_result = {
                    "imageID": image_id,
                    "success": True,
                    "error": None,
                    "bgReplacedTemporary": image_urls["public_url"],
                    "bgReplacedPermanent": image_urls["permanent_url"],
                    "steps": [
                        {"step": 1, "action": "background_remove", "aiModels": "RMBG-2.0", "methods": None, "version": 2.0, "elapsedSeconds": round(rmbg_time, 2)},
                        {"step": 2, "action": "insert_background", "aiModels": None, "methods": "alpha_composite", "version": None, "elapsedSeconds": round(comp_time, 2)},
                    ]
                }
                all_image_results.append(operation_result)

            except Exception as e:
                print(f"ERROR: Background replacement failed for image {image_id}: {e}")
                all_image_results.append({
                    "imageID": image_id, "success": False, "error": str(e),
                    "bgReplacedTemporary": None, "bgReplacedPermanent": None, "steps": []
                })

        final_output_data["results"] = all_image_results
        total_process_time = time.perf_counter() - processing_start_time

        # 3. Finalization
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

        # 4. Prune Output
        if output_verbosity == "default":
            final_output_data.pop("steps", None)
            pruned_results = []
            keys_to_include = {"imageID", "bgReplacedTemporary", "bgReplacedPermanent"}
            for result in final_output_data["results"]:
                if result.get("success"):
                    pruned_results.append({key: result[key] for key in keys_to_include if key in result})
                else:
                    pruned_results.append(result)
            final_output_data["results"] = pruned_results
        
        return {"success": True, "output": final_output_data}

    except Exception as e:
        print(f"FATAL PIPELINE ERROR: Background replacement task {task_id} failed: {e}")
        error_report = {
            "parentID": request_data.get("parent_id"),
            "taskID": task_id,
            "status": "FAILURE",
            "error": str(e)
        }
        azure_service.upload_json(data=error_report, file_name=f"{task_id}_error.json", is_input=False)
        raise
