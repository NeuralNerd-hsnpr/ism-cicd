# tasks/treatment_task.py

import time
import uuid
import httpx
from io import BytesIO
from PIL import Image

# Import Celery app and settings
from tasks.celery_app import celery_app, settings

# Import all the new service modules we've created
from services import azure_service, api_service, llm_service, image_utils
from services.api_service import AIRequestError

from services.llm_service import LLMServiceError


# --- Custom Exceptions for Clear Error Handling ---
class TreatmentOrchestrationError(Exception):
    """Custom exception for failures during the treatment orchestration."""
    pass

# This is the custom sorting order from your original code, which we will preserve.
OPERATION_ORDER = {"REMOVE": 0, "REPLACE": 1, "ADJUST": 2, "PLACE": 3, "FILTER": 4, "STYLE": 5}

# This hard-coded dictionary of known object sizes is from your original code.
# It's used to avoid unnecessary LLM calls.
KNOWN_OBJECT_SIZES_CM = {
    "hands": [12, 5], "hand": [12, 5], "face": [12, 18], "head": [14, 22],
    "eyes": [12, 3], "nose": [2, 4], "mouth": [5, 2], "ear": [3, 6], "hair": [14, 8],
    "body": [50, 165], "cat": [40, 20], "dog": [40, 20], "ball": [5, 5],
    "baseball" : [5,5], "coca cola can": [5, 14], "sword":[10,80],
    "cap":[20,7], "nightcap":[20,7], "hat":[20,7], "shirt": [40, 60], "pants": [40, 100],
}

@celery_app.task(bind=True)
def process_complex_treatment_request(self, request_data: dict):
    """
    Orchestrates the entire treatment pipeline by calling various utility
    and AI microservices.
    """
    task_id = self.request.id
    parent_id = request_data.get("parent_id", str(uuid.uuid4()))
    print(f"Starting treatment task {task_id} for parentID {parent_id}")

    try:
        # --- 1. Initialization ---
        full_start_time = time.perf_counter()
        output_verbosity = request_data.get("output", "default").lower()
        final_output_data = {"parentID": parent_id, "elapsedSeconds": 0, "steps": [], "results": []}
        azure_service.upload_json(data=request_data, file_name=f"{task_id}_input.json", is_input=True)
        upload_input_time = time.perf_counter() - full_start_time

        # --- 2. Main Processing Loop for Each Image ---
        processing_start_time = time.perf_counter()
        
        for image_data in request_data.get("images", []):
            image_id = image_data.get("image_id")
            base_image_url = image_data.get("base_url")
            
            # This state dictionary tracks the progress for a single image.
            image_state = {
                "checkpoints": [image_utils.load_image_from_url(base_image_url)],
                "timings": {"detection": 0, "llm": 0, "flux": 0, "removal": 0, "style": 0, "filter": 0, "network": 0, "adjust_bbox": 0},
                "final_urls": {},
                "operation_descriptions": [],
                "adjusted_mask_urls": None, 
                "treatment_phrase": None,
                "cap_pose": None,
            }

            # Sort operations based on the required execution order.
            operations = sorted(image_data.get("operations", []), key=lambda op: OPERATION_ORDER.get(op.get("type", ""), 99))

            for op in operations:
                op_type = op.get("type")
                print(f"Orchestrating operation '{op_type}' for image '{image_id}'")

                # --- MODIFIED: Call helper functions with full context ---
                if op_type == "REMOVE":
                    _handle_remove_operation(op, image_state, image_id, parent_id)
                elif op_type in ["REPLACE", "ADJUST"]:
                    _handle_replace_adjust_operation(op, image_state, image_id, parent_id)
                elif op_type == "PLACE":
                    _handle_place_operation(op, image_state, image_id, parent_id)
                elif op_type == "FILTER":
                    _handle_filter_operation(op, image_state) # No parent/image ID needed
                elif op_type == "STYLE":
                    _handle_style_operation(op, image_state) # No parent/image ID needed

            # After all operations, save the final image.
            final_image_pil = image_state["checkpoints"][-1]
            final_image_bytes = BytesIO()
            final_image_pil.save(final_image_bytes, format='PNG')
            
            image_state["final_urls"] = azure_service.save_image_locally_and_get_urls(
                image_bytes=final_image_bytes.getvalue(),
                file_name=f"{task_id}_{image_id}_treatment_result.png"
            )

            final_output_data["results"].append(_build_image_result_dict(image_id, base_image_url, image_state, output_verbosity))

        total_process_time = time.perf_counter() - processing_start_time

        # --- 3. Finalization ---
        # ... (Build final steps and upload output JSON) ...
        # (This logic is very similar to the inpainting task)
        
        return {"success": True, "output": final_output_data}

    except Exception as e:
        print(f"FATAL PIPELINE ERROR in treatment task {task_id}: {e}")
        azure_service.upload_json(data={"error": str(e)}, file_name=f"{task_id}_error.json", is_input=False)
        raise  # Re-raise to mark task as FAILED

# ===================================================================
# PRIVATE HELPER FUNCTIONS FOR EACH OPERATION TYPE
# ===================================================================

def _handle_remove_operation(op: dict, state: dict, image_id: str, parent_id: str):
    """
    Handles the REMOVE operation by getting a mask (via detection if needed)
    and calling the Lama removal microservice.
    """
    start_time = time.perf_counter()
    current_image_pil = state["checkpoints"][-1]
    detect_phrase = op.get("source_detected_code")
    
    mask_url = op.get("source_detected_mask_url")
    
    try:
        if not mask_url:
            print(f"No mask provided for REMOVE. Calling detection service for '{detect_phrase}'...")
            detection_start_time = time.perf_counter()
            
            # Convert current PIL image to bytes to send to detection
            image_bytes_for_detection = BytesIO()
            current_image_pil.save(image_bytes_for_detection, format='PNG')
            
            mask_url = api_service.call_detection_service(
                parent_id=parent_id,
                image_id=image_id,
                image_bytes=image_bytes_for_detection.getvalue(),
                detect_phrase=detect_phrase
            )
            state["timings"]["detection"] += time.perf_counter() - detection_start_time

        # Load the mask from its URL and process it
        network_start_time = time.perf_counter()
        mask_pil = image_utils.load_image_from_url(mask_url)
        state["timings"]["network"] += time.perf_counter() - network_start_time
        
        # Apply the bounding box logic from your original code
        bbox_mask_pil = image_utils.get_bounding_box_masked_image(mask_pil)
        
        # Convert PIL images to bytes for the Lama API call
        image_bytes = BytesIO(); current_image_pil.save(image_bytes, format='PNG')
        mask_bytes = BytesIO(); bbox_mask_pil.save(mask_bytes, format='PNG')

        # Call the Lama removal microservice via our new, robust client
        removal_start_time = time.perf_counter()
        result_bytes = api_service.call_lama_service(image_bytes.getvalue(), mask_bytes.getvalue())
        state["timings"]["removal"] += time.perf_counter() - removal_start_time
        
        # Update the pipeline state with the new image
        new_image = Image.open(BytesIO(result_bytes))
        state["checkpoints"].append(new_image)
        state["operation_descriptions"].append(f"removed {detect_phrase}")

    except AIRequestError as e:
        # If any downstream AI service fails, wrap it in our orchestration error
        # so the main task can fail gracefully.
        raise TreatmentOrchestrationError(f"Failed during REMOVE operation: {e}") from e

def _handle_replace_adjust_operation(op: dict, state: dict, image_id: str, parent_id: str):
    """
    Handles both REPLACE and ADJUST operations by preparing the appropriate mask
    and then calling the unified Flux AI client.
    """
    op_type = op.get("type")
    print(f"Handling {op_type} operation for image '{image_id}'...")
    start_time = time.perf_counter()
    current_image_pil = state["checkpoints"][-1]
    
    try:
        # --- 1. Get the Source Mask (calling detection service if necessary) ---
        mask_url = op.get("source_detected_mask_url")
        if not mask_url:
            detect_phrase = op.get("source_detected_code")
            print(f"No mask provided for {op_type}. Calling detection service for '{detect_phrase}'...")
            detection_start_time = time.perf_counter()
            image_bytes_for_detection = BytesIO(); current_image_pil.save(image_bytes_for_detection, format='PNG')
            mask_url = api_service.call_detection_service(
                parent_id=parent_id, image_id=image_id,
                image_bytes=image_bytes_for_detection.getvalue(), detect_phrase=detect_phrase
            )
            state["timings"]["detection"] += time.perf_counter() - detection_start_time

        # --- 2. Load and Process Masks ---
        network_start_time = time.perf_counter()
        mask_pil = image_utils.load_image_from_url(mask_url)
        state["timings"]["network"] += time.perf_counter() - network_start_time
        
        # This is the key difference: REPLACE uses the bounding box of the mask.
        processed_mask = image_utils.get_bounding_box_masked_image(mask_pil.convert("L")) if op_type == "REPLACE" else mask_pil.convert("L")

        # Handle avoidance masks if they are provided
        final_mask = processed_mask
        avoidance_urls = op.get("avoidance_mask_urls")
        if avoidance_urls:
            print("Filtering mask with avoidance masks...")
            network_start_time = time.perf_counter()
            avoidance_masks_pil = [image_utils.load_image_from_url(url) for url in avoidance_urls]
            state["timings"]["network"] += time.perf_counter() - network_start_time
            final_mask = image_utils.filter_mask(base_mask=processed_mask, filter_masks=avoidance_masks_pil)

        # Save the final adjusted mask for debugging and reporting
        highlight_image = image_utils.create_transparent_highlight(current_image_pil, final_mask)
        mask_bytes_for_upload = BytesIO(); highlight_image.save(mask_bytes_for_upload, format='PNG')
        state["adjusted_mask_urls"] = azure_service.save_image_locally_and_get_urls(
            image_bytes=mask_bytes_for_upload.getvalue(),
            file_name=f"{parent_id}_{image_id}_{op.get('source_detected_code')}_adjusted_mask.png"
        )
        
        # --- 3. Prepare Data and Call the AI Client (Flux or SDXL) ---
        treatment_prompt = op.get("treatment")
        params = op.get("model_parameters", {})
        
        # Determine which model to use (default to Flux)
        model_choice = (op.get("model") or "flux").lower()
        
        current_image_bytes = BytesIO(); current_image_pil.save(current_image_bytes, format='PNG')
        final_mask_bytes = BytesIO(); final_mask.save(final_mask_bytes, format='PNG')
        
        flux_start_time = time.perf_counter()
        
        if "sdxl" in model_choice:
             print(f"Using SDXL Inpainting for {op_type}...")
             result_bytes = api_service.call_sdxl_inpainting_service(
                image_bytes=current_image_bytes.getvalue(),
                mask_bytes=final_mask_bytes.getvalue(),
                prompt=treatment_prompt,
                params=params
            )
        else:
            print(f"Using Flux Fill for {op_type}...")
            flux_params = {
                'num_inference_steps': params.get("inference_steps", 35),
                'guidance_scale': params.get("guidance_scale", 30),
                'seed': params.get("seed", 0) if not params.get("randomized_seed", True) else None
            }
            result_bytes = api_service.call_flux_service(
                image_bytes=current_image_bytes.getvalue(),
                mask_bytes=final_mask_bytes.getvalue(),
                prompt=treatment_prompt,
                params=flux_params
            )
            
        state["timings"]["flux"] += time.perf_counter() - flux_start_time

        # --- 4. Update State ---
        new_image = Image.open(BytesIO(result_bytes))
        state["checkpoints"].append(new_image)
        state["operation_descriptions"].append(f"{op_type.lower()}d {op.get('source_detected_code')} with {treatment_prompt}")

    except AIRequestError as e:
        # Catch errors from our AI clients and fail the task gracefully
        raise TreatmentOrchestrationError(f"Failed during {op_type} operation: {e}") from e


def _handle_place_operation(op: dict, state: dict, image_id: str, parent_id: str):
    """
    Handles the complex PLACE operation by orchestrating calls to LLM, detection,
    and image utility services before calling the final inpainting service
    via the unified AI client.
    """
    start_time = time.perf_counter()
    current_image_pil = state["checkpoints"][-1]
    
    try:
        # --- 1. Extract All Parameters from the Operation Dictionary ---
        placement_type = op.get("placement_type", "NEAR")
        position = op.get("location", "center").lower()
        detect_phrase = op.get("source_detected_code", "").lower()
        treatment_prompt = op.get("treatment")
        
        # --- 2. Determine Treatment Target Size (using LLM if needed) ---
        treatment_target_size = op.get("treatment_target_size")
        llm_start_time = time.perf_counter()
        treatment_phrase = llm_service.extract_main_object_name(treatment_prompt)
        state["treatment_phrase"] = treatment_phrase
        if not treatment_target_size:
            if treatment_phrase in KNOWN_OBJECT_SIZES_CM:
                sizes = KNOWN_OBJECT_SIZES_CM[treatment_phrase]
                treatment_target_size = [sizes[1], sizes[0]] # [height, width]
                print(f"Used known size for '{treatment_phrase}': {treatment_target_size}")
            else:
                print(f"Estimating size for '{treatment_phrase}' via LLM...")
                sizes_dict = llm_service.estimate_object_dimensions(treatment_phrase)
                treatment_target_size = [sizes_dict["actual_height"], sizes_dict["actual_width"]]
        state["timings"]["llm"] += time.perf_counter() - llm_start_time

        # --- 3. Determine Source Object Size (for pixel density calculation) ---
        actual_width_cm, actual_height_cm = None, None
        if op.get("detected_object_pixels_per_cm") is None:
            llm_start_time = time.perf_counter()
            if detect_phrase in KNOWN_OBJECT_SIZES_CM:
                sizes = KNOWN_OBJECT_SIZES_CM[detect_phrase]
                actual_width_cm, actual_height_cm = sizes[0], sizes[1]
                print(f"Used known size for '{detect_phrase}': {[actual_height_cm, actual_width_cm]}")
            else:
                print(f"Estimating size for '{detect_phrase}' via LLM...")
                sizes_dict = llm_service.estimate_object_dimensions(detect_phrase)
                actual_width_cm, actual_height_cm = sizes_dict["actual_width"], sizes_dict["actual_height"]
            state["timings"]["llm"] += time.perf_counter() - llm_start_time

        # --- 4. Get and Process Source Mask ---
        mask_url = op.get("source_detected_mask_url")
        if not mask_url:
            print(f"No mask provided for PLACE. Calling detection service for '{detect_phrase}'...")
            detection_start_time = time.perf_counter()
            image_bytes_for_detection = BytesIO(); current_image_pil.save(image_bytes_for_detection, format='PNG')
            mask_url = api_service.call_detection_service(
                parent_id=parent_id, image_id=image_id,
                image_bytes=image_bytes_for_detection.getvalue(), detect_phrase=detect_phrase
            )
            state["timings"]["detection"] += time.perf_counter() - detection_start_time
        
        network_start_time = time.perf_counter()
        detected_mask_pil = image_utils.load_image_from_url(mask_url)
        state["timings"]["network"] += time.perf_counter() - network_start_time
        bbox_mask_pil = image_utils.get_bounding_box_masked_image(detected_mask_pil.convert("L"))

        # --- 5. Handle Avoidance Masks and Special Scenarios ---
        avoidance_masks_pil = []
        if op.get("avoidance_mask_urls"):
            network_start_time = time.perf_counter()
            avoidance_masks_pil = [image_utils.load_image_from_url(url) for url in op["avoidance_mask_urls"]]
            state["timings"]["network"] += time.perf_counter() - network_start_time
        
        state["cap_pose"] = None 
        if treatment_phrase in ["cap", "hat", "nightcap"]:
            print(f"Detecting head pose for '{treatment_phrase}' placement...")
            try:
                # Prepare images for Gaze service
                gaze_base_bytes = BytesIO()
                current_image_pil.save(gaze_base_bytes, format='PNG')
                
                gaze_mask_bytes = BytesIO()
                bbox_mask_pil.save(gaze_mask_bytes, format='PNG') 
                
                gaze_start = time.perf_counter()
                direction = api_service.call_gaze_service(
                    base_image_bytes=gaze_base_bytes.getvalue(),
                    mask_image_bytes=gaze_mask_bytes.getvalue()
                )
                state["timings"]["network"] += time.perf_counter() - gaze_start
                
                state["cap_pose"] = direction
                print(f"Detected head pose: {direction}")
                
            except Exception as e:
                print(f"WARNING: Gaze detection failed: {e}. Defaulting to None.")
                state["cap_pose"] = None

        # --- 6. Resize the Placement Mask ---
        bbox_adjust_start = time.perf_counter()
        adjusted_mask = image_utils.resize_mask_rectangle(
            mask_image=bbox_mask_pil,
            object_width_cm=treatment_target_size[1],
            object_height_cm=treatment_target_size[0],
            position=position,
            detected_object_pixels_per_cm=op.get("detected_object_pixels_per_cm"),
            actual_width_cm=actual_width_cm,
            actual_height_cm=actual_height_cm,
            placement_rotation=op.get("placement_rotation"),
            include_original=(placement_type == "WITHIN"),
            behind=(placement_type == "BEHIND"),
            cap_pose=state["cap_pose"]
        )
        state["timings"]["adjust_bbox"] += time.perf_counter() - bbox_adjust_start

        # --- 7. Finalize Mask ---
        final_mask = adjusted_mask
        if placement_type == "BEHIND":
            final_mask = image_utils.subtract_masks(adjusted_mask, detected_mask_pil)
        if avoidance_masks_pil:
            final_mask = image_utils.filter_mask(base_mask=final_mask, filter_masks=avoidance_masks_pil)
        
        # Save the final adjusted mask for debugging/output
        highlight_image = image_utils.create_transparent_highlight(current_image_pil, final_mask)
        mask_bytes_for_upload = BytesIO(); highlight_image.save(mask_bytes_for_upload, format='PNG')
        state["adjusted_mask_urls"] = azure_service.save_image_locally_and_get_urls(
            image_bytes=mask_bytes_for_upload.getvalue(),
            file_name=f"{parent_id}_{image_id}_{detect_phrase}_adjusted_mask.png"
        )
        
        # --- 8. Call Inpainting Service (Flux or SDXL) ---
        params = op.get("model_parameters", {})
        model_choice = (op.get("model") or "flux").lower()

        current_image_bytes = BytesIO(); current_image_pil.save(current_image_bytes, format='PNG')
        final_mask_bytes = BytesIO(); final_mask.save(final_mask_bytes, format='PNG')

        flux_start_time = time.perf_counter()
        
        if "sdxl" in model_choice:
             print(f"Using SDXL Inpainting for PLACE...")
             result_bytes = api_service.call_sdxl_inpainting_service(
                image_bytes=current_image_bytes.getvalue(),
                mask_bytes=final_mask_bytes.getvalue(),
                prompt=treatment_prompt,
                params=params
            )
        else:
            print(f"Using Flux Fill for PLACE...")
            flux_params = {
                'num_inference_steps': params.get("inference_steps"),
                'guidance_scale': params.get("guidance_scale"),
                'seed': params.get("seed") if not params.get("randomized_seed", True) else None
            }
            result_bytes = api_service.call_flux_service(
                image_bytes=current_image_bytes.getvalue(),
                mask_bytes=final_mask_bytes.getvalue(),
                prompt=treatment_prompt,
                params=flux_params
            )
            
        state["timings"]["flux"] += time.perf_counter() - flux_start_time

        # --- 9. Update State ---
        new_image = Image.open(BytesIO(result_bytes))
        state["checkpoints"].append(new_image)
        state["operation_descriptions"].append(f"placed {treatment_prompt} on {position} of {detect_phrase}")
        
    except (AIRequestError, LLMServiceError) as e:
        raise TreatmentOrchestrationError(f"Failed during PLACE operation: {e}") from e

def _handle_filter_operation(op: dict, state: dict):
    """
    Handles the FILTER operation by delegating the work to the image_utils service.
    This function remains clean and free of complex if/elif logic.
    """
    start_time = time.perf_counter()
    current_image_pil = state["checkpoints"][-1]
    filter_type = op.get("filter_type")

    if not filter_type:
        print("WARNING: 'filter_type' not specified in FILTER operation. Skipping.")
        # We append the current image to maintain the checkpoint chain
        state["checkpoints"].append(current_image_pil)
        return

    try:
        # --- The core logic is a single, clean function call ---
        new_image = image_utils.apply_filter(
            image_pil=current_image_pil,
            filter_type=filter_type
        )
        
        state["timings"]["filter"] += time.perf_counter() - start_time
        
        # Update the state with the new, filtered image
        state["checkpoints"].append(new_image)
        state["operation_descriptions"].append(f"applied {filter_type} filter")

    except Exception as e:
        # Catch any unexpected errors from the image processing library
        print(f"ERROR: An unexpected error occurred during the '{filter_type}' filter: {e}")
        raise TreatmentOrchestrationError(f"Failed during FILTER operation: {e}") from e

def _handle_style_operation(op: dict, state: dict):
    """
    Handles the STYLE operation by calling the SD Styling microservice
    via the unified AI client, correctly handling model selection.
    """
    start_time = time.perf_counter()
    current_image_pil = state["checkpoints"][-1]
    style_prompt = op.get("style_description")

    # 1. Determine which model the user requested for styling.
    # This logic is adapted from your original code.
    model_choice = (op.get("model") or "stable_diffusion_controlnet").lower()
    
    # 2. Map the long model name from the input JSON to the short name
    #    that our new /style API endpoint expects ('pix2pix' or 'controlnet').
    if "controlnet" in model_choice:
        api_model = "controlnet"
    elif "pix2pix" in model_choice:
        api_model = "pix2pix"
    else:
        # Provide a safe fallback if an unknown model name is provided in the future.
        print(f"WARNING: Unknown style model '{model_choice}'. Defaulting to 'controlnet'.")
        api_model = "controlnet"

    try:
        # 3. Convert the current PIL image to bytes for the API call.
        image_bytes_for_styling = BytesIO()
        current_image_pil.save(image_bytes_for_styling, format='PNG')
        
        # 4. Call the SD Styling microservice via the unified AI client.
        styling_start_time = time.perf_counter()
        result_bytes = api_service.call_style_service(
            image_bytes=image_bytes_for_styling.getvalue(),
            style_prompt=style_prompt,
            model=api_model  # Pass the correctly mapped model name.
        )
        state["timings"]["style"] += time.perf_counter() - styling_start_time
        
        # 5. Update the state with the new, stylized image.
        new_image = Image.open(BytesIO(result_bytes))
        state["checkpoints"].append(new_image)
        state["operation_descriptions"].append(f"applied style: {style_prompt}")

    except AIRequestError as e:
        # Catch errors from our AI client and fail the task gracefully.
        raise TreatmentOrchestrationError(f"Failed during STYLE operation: {e}") from e

def _build_image_result_dict(image_id: str, base_url: str, state: dict, verbosity: str) -> dict:
    """Builds the final 'results' entry for a single image, respecting verbosity."""
    
    # This mirrors the 'runtimes' structure from your old code
    verbose_steps = [
        {"step": 1, "action": "retrieve mask image", "aiModels": "detection service API call", "elapsedSeconds": round(state["timings"]["detection"], 2)},
        {"step": 2, "action": "retrieve actual sizes and treatment code", "aiModels": "LLM Service", "elapsedSeconds": round(state["timings"]["llm"], 2)},
        {"step": 3, "action": "mask adjustment", "methods": "opencv_pillow_numpy", "elapsedSeconds": round(state["timings"]["adjust_bbox"], 2)},
        {"step": 4, "action": "in-painting", "aiModels": "flux_fill_dev", "elapsedSeconds": round(state["timings"]["flux"], 2)},
        {"step": 5, "action": "object removal", "aiModels": "simple_lama_inpainting", "elapsedSeconds": round(state["timings"]["removal"], 2)},
        {"step": 6, "action": "styling", "aiModels": "stable_diffusion_controlnet", "elapsedSeconds": round(state["timings"]["style"], 2)},
        {"step": 7, "action": "filtering", "methods": "opencv_numpy", "elapsedSeconds": round(state["timings"]["filter"], 2)},
        {"step": 8, "action": "total_network_runtime", "methods": "requests", "elapsedSeconds": round(state["timings"]["network"], 2)},
    ]

    full_result = {
        "imageID": image_id,
        "success": True,
        "error": None,
        "baseUrl": base_url,
        "adjustedMaskTemporary": state["adjusted_mask_urls"].get("public_url") if state["adjusted_mask_urls"] else None,
        "adjustedMaskPermanent": state["adjusted_mask_urls"].get("permanent_url") if state["adjusted_mask_urls"] else None,
        "imageTemporary": state["final_urls"].get("public_url"),
        "imagePermanent": state["final_urls"].get("permanent_url"),
        "treatmentCode": state["treatment_phrase"],
        "capPose": state["cap_pose"],
        "operationsDescription": {f"operation {i+1}": desc for i, desc in enumerate(state["operation_descriptions"])},
        "steps": verbose_steps
    }

    if verbosity == "default":
        keys_to_include = {"imageID", "baseUrl", "adjustedMaskTemporary", "adjustedMaskPermanent", "imageTemporary", "imagePermanent"}
        return {key: full_result[key] for key in keys_to_include if key in full_result}
    
    return full_result