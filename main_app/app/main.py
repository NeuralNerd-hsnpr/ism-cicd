from fastapi import FastAPI, HTTPException
from celery.result import AsyncResult
from tasks.celery_app import celery_app
from services import azure_service
import asyncio
from services.url_validator import validate_image_urls_are_accessible
from tasks.inpainting_task import process_complex_inpainting_request
from tasks.treatment_task import process_complex_treatment_request
from app.schemas import (
    ComplexInpaintingRequest,
    ComplexTreatmentRequest,
    TaskCreationResponse,
    ComplexInpaintingResponse,
    TaskSummaryResponse,
    TaskFailureResponse,
    ProcessingMethod,
    BackgroundRemovalRequest,
    BackgroundRemovalResponse,
    BackgroundInsertionRequest,
    BackgroundReplacementRequest,
    VideoGenerationRequest
)
from tasks.background_task import (
    process_background_removal_request,
    process_background_insertion_request,
    process_background_replacement_request
)

app = FastAPI(title="User-Facing Inpainting Service")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/inpainting",
          # The `responses` dictionary is the best way to document multiple
          # possible success responses in Swagger/OpenAPI.
          responses={
              200: {"model": ComplexInpaintingResponse, "description": "Successful synchronous response with full output."},
              202: {"model": TaskCreationResponse, "description": "Request accepted for asynchronous processing."},
              400: {"description": "Invalid input, such as an inaccessible image URL."},
              500: {"model": TaskFailureResponse, "description": "The background task failed during synchronous execution."}
          })
async def start_inpainting(request: ComplexInpaintingRequest):
    """
    Accepts a complex inpainting request for processing.

    This endpoint supports two processing methods controlled by the 'method' field:
    - **async (default):** The request is immediately queued for background
      processing, and a task ID is returned. The client must poll the
      /results/{task_id} or /task_summary/{task_id} endpoint to get the final output.
      
    - **sync:** The request is queued with high priority, and the API will wait
      for the processing to complete before sending a response. The full output
      is returned directly in the response body.
    """
    # 1. Pre-emptive Validation:
    # Before queueing, validate that all image URLs are live and accessible.
    # This provides immediate feedback to the user for invalid input.
    # If a URL is bad, this function will raise an HTTPException and stop execution.
    await validate_image_urls_are_accessible(request.images)
    
    # Convert the Pydantic model to a JSON-serializable dictionary for Celery.
    task_data = request.model_dump(mode='json')
    
    # 2. Branch Logic based on the requested method.
    if request.method == ProcessingMethod.ASYNC:
        # ASYNCHRONOUS WORKFLOW: Queue and return immediately.
        print(f"Handling request for parentID '{request.parent_id}' as ASYNC.")
        
        # Use .apply_async to specify the queue.
        task = process_complex_inpainting_request.apply_async(
            args=[task_data],
            queue="inpainting_async"  # Send to the standard, low-priority queue.
        )
        
        # Return a 202 Accepted response with the task ID.
        return {"task_id": task.id}

    else: # SYNCHRONOUS WORKFLOW: Queue, wait, and return the full result.
        print(f"Handling request for parentID '{request.parent_id}' as SYNC.")
        
        # Queue the task with high priority.
        task = process_complex_inpainting_request.apply_async(
            args=[task_data],
            queue="inpainting_sync"  # Send to the high-priority queue.
        )
        
        try:
            # Wait for the result without blocking the FastAPI event loop.
            # asyncio.to_thread runs the blocking task.get() in a separate thread.
            # A timeout is a crucial safeguard against unexpectedly long tasks.
            print(f"Waiting for sync task {task.id} to complete...")
            result = await asyncio.to_thread(task.get, timeout=300) # 5-minute timeout
            print(f"Sync task {task.id} completed.")

        except Exception as e:
            # This catches Celery timeouts or other errors during the waiting period.
            raise HTTPException(
                status_code=504, # Gateway Timeout
                detail=f"Synchronous request timed out or failed while waiting for the result: {e.__class__.__name__}"
            )

        # After getting the result, check if the task itself failed.
        if task.failed():
            # The task raised an exception. Return a 500 error with the failure details.
            raise HTTPException(
                status_code=500, # Internal Server Error
                detail={
                    "taskId": task.id,
                    "status": "FAILURE",
                    "errorInfo": str(task.info) # task.info holds the exception string.
                }
            )
        
        # If the task succeeded, return the full result with a 200 OK status.
        return result



@app.post("/treatment",
          responses={
              200: {"description": "Successful synchronous response with full output."}, # Define a response model later
              202: {"model": TaskCreationResponse, "description": "Request accepted for asynchronous processing."},
              400: {"description": "Invalid input, such as an inaccessible image URL."},
              500: {"model": TaskFailureResponse, "description": "The background task failed during execution."}
          })
async def start_treatment(request: ComplexTreatmentRequest):
    """
    Accepts a complex treatment request with various operation types.
    Supports both 'async' and 'sync' processing methods.
    """
    # NOTE: You will need to create a new URL validator for this complex structure.
    # For now, we will skip this step to get the endpoint working.
    # await validate_treatment_urls_are_accessible(request.images)
    
    task_data = request.model_dump(mode='json')
    
    if request.method == ProcessingMethod.ASYNC:
        print(f"Handling treatment request for parentID '{request.parent_id}' as ASYNC.")
        task = process_complex_treatment_request.apply_async(
            args=[task_data],
            queue="treatment_async"
        )
        return {"task_id": task.id}

    else: # SYNC
        print(f"Handling treatment request for parentID '{request.parent_id}' as SYNC.")
        task = process_complex_treatment_request.apply_async(
            args=[task_data],
            queue="treatment_sync"
        )
        
        try:
            result = await asyncio.to_thread(task.get, timeout=300)
        except Exception as e:
            raise HTTPException(status_code=504, detail=f"Sync request timed out or failed: {e}")

        if task.failed():
            raise HTTPException(status_code=500, detail={"taskId": task.id, "status": "FAILURE", "errorInfo": str(task.info)})
        
        return result


@app.post("/remove-background",
          responses={
              200: {"model": ComplexInpaintingResponse, "description": "Successful synchronous response with full output."}, # Reusing response model for now, or create specific one
              202: {"model": TaskCreationResponse, "description": "Request accepted for asynchronous processing."},
              400: {"description": "Invalid input."},
              500: {"model": TaskFailureResponse, "description": "The background task failed during execution."}
          })
async def remove_background(request: BackgroundRemovalRequest):
    """
    Accepts a background removal request.
    Supports both 'async' and 'sync' processing methods.
    """
    # await validate_image_urls_are_accessible(request.images) # TODO: Adapt validator for this schema if needed

    task_data = request.model_dump(mode='json')

    if request.method == ProcessingMethod.ASYNC:
        print(f"Handling background removal request for parentID '{request.parent_id}' as ASYNC.")
        task = process_background_removal_request.apply_async(
            args=[task_data],
            queue="background_async"
        )
        return {"task_id": task.id}

    else: # SYNC
        print(f"Handling background removal request for parentID '{request.parent_id}' as SYNC.")
        task = process_background_removal_request.apply_async(
            args=[task_data],
            queue="background_sync"
        )
        
        try:
            result = await asyncio.to_thread(task.get, timeout=300)
        except Exception as e:
            raise HTTPException(status_code=504, detail=f"Sync request timed out or failed: {e}")

        if task.failed():
            raise HTTPException(status_code=500, detail={"taskId": task.id, "status": "FAILURE", "errorInfo": str(task.info)})
        
        return result


@app.post("/insert-background",
          responses={
              200: {"model": ComplexInpaintingResponse, "description": "Successful synchronous response."},
              202: {"model": TaskCreationResponse, "description": "Request accepted for asynchronous processing."},
              400: {"description": "Invalid input."},
              500: {"model": TaskFailureResponse, "description": "Task failed."}
          })
async def insert_background(request: BackgroundInsertionRequest):
    """
    Accepts a background insertion request.
    Input URL should be a transparent image (PNG).
    """
    task_data = request.model_dump(mode='json')

    if request.method == ProcessingMethod.ASYNC:
        print(f"Handling background insertion request for parentID '{request.parent_id}' as ASYNC.")
        task = process_background_insertion_request.apply_async(
            args=[task_data],
            queue="background_async"
        )
        return {"task_id": task.id}

    else: # SYNC
        print(f"Handling background insertion request for parentID '{request.parent_id}' as SYNC.")
        task = process_background_insertion_request.apply_async(
            args=[task_data],
            queue="background_sync"
        )
        
        try:
            result = await asyncio.to_thread(task.get, timeout=300)
        except Exception as e:
            raise HTTPException(status_code=504, detail=f"Sync request timed out or failed: {e}")

        if task.failed():
            raise HTTPException(status_code=500, detail={"taskId": task.id, "status": "FAILURE", "errorInfo": str(task.info)})
        
        return result


@app.post("/replace-background",
          responses={
              200: {"model": ComplexInpaintingResponse, "description": "Successful synchronous response."},
              202: {"model": TaskCreationResponse, "description": "Request accepted for asynchronous processing."},
              400: {"description": "Invalid input."},
              500: {"model": TaskFailureResponse, "description": "Task failed."}
          })
async def replace_background(request: BackgroundReplacementRequest):
    """
    Accepts a background replacement request.
    Removes background from input URL, then inserts the new background.
    """
    task_data = request.model_dump(mode='json')

    if request.method == ProcessingMethod.ASYNC:
        print(f"Handling background replacement request for parentID '{request.parent_id}' as ASYNC.")
        task = process_background_replacement_request.apply_async(
            args=[task_data],
            queue="background_async"
        )
        return {"task_id": task.id}

    else: # SYNC
        print(f"Handling background replacement request for parentID '{request.parent_id}' as SYNC.")
        task = process_background_replacement_request.apply_async(
            args=[task_data],
            queue="background_sync"
        )
        
        try:
            result = await asyncio.to_thread(task.get, timeout=300)
        except Exception as e:
            raise HTTPException(status_code=504, detail=f"Sync request timed out or failed: {e}")

        if task.failed():
            raise HTTPException(status_code=500, detail={"taskId": task.id, "status": "FAILURE", "errorInfo": str(task.info)})
        
        return result


@app.post("/generate-video", response_model=TaskCreationResponse)
async def generate_video(request: VideoGenerationRequest):
    """
    Initiates an asynchronous video generation task (Text-to-Video or Image-to-Video).
    """
    # Send to Celery
    task = celery_app.send_task(
        "tasks.video_task.process_video_generation_request",
        args=[request.model_dump(by_alias=True)],
        queue="video_async"
    )
    return TaskCreationResponse(task_id=task.id)


@app.get("/results/{task_id}", 
         response_model=ComplexInpaintingResponse,
         responses={500: {"model": TaskFailureResponse}})
def get_task_result(task_id: str):
    """
    Fetches the result of a completed task.
    Returns the full success payload if the task succeeded,
    or a 500 error with failure details if the task failed.
    """
    task_result = AsyncResult(task_id, app=celery_app)

    if not task_result.ready():
        raise HTTPException(status_code=202, detail="Task is still processing.")

    if task_result.failed():
        # The task failed, so we return a structured error response
        # with a 500 status code to indicate a server-side failure.
        raise HTTPException(status_code=500, detail={
            "taskId": task_id,
            "status": "FAILURE",
            "errorInfo": str(task_result.info) # .info holds the exception
        })

    # If we reach here, the task was successful.
    return task_result.get()



@app.get("/task_summary/{task_id}", response_model=TaskSummaryResponse)
async def get_task_summary(task_id: str):
    """
    Retrieves a comprehensive summary for a given task ID, including its
    current status, input payload, and final output or error information.
    """
    # 1. Get the task's current state from Celery's result backend
    task_result = AsyncResult(task_id, app=celery_app)
    task_status = task_result.status

    # 2. Define the filenames based on the task_id
    input_filename = f"{task_id}_input.json"
    output_filename = f"{task_id}_output.json"

    # 3. Concurrently fetch the input and output JSONs from Azure
    print(f"Concurrently fetching payloads for task: {task_id}")
    try:
        print(f"API DOWNLOADER: Attempting to download with filenames: '{input_filename}', '{output_filename}'")
        # asyncio.gather runs both awaitable functions in parallel
        input_payload, output_payload = await asyncio.gather(
            azure_service.download_json_async(file_name=input_filename, is_input=True),
            azure_service.download_json_async(file_name=output_filename, is_input=False)
        )
    except Exception as e:
        # Handle potential connection errors with Azure itself
        raise HTTPException(status_code=503, detail=f"Failed to communicate with Azure Storage: {e}")

    # 4. Compile the final response
    response_data = {
        "taskId": task_id,
        "status": task_status,
        "inputPayload": input_payload,
        "outputPayload": output_payload,
        "errorInfo": None
    }

    if task_result.failed():
        # If the task failed, get the error information from Celery
        response_data["errorInfo"] = str(task_result.info) # .info usually holds the exception

    elif task_status == "SUCCESS" and output_payload is None:
        # Edge case: Task succeeded but output file is missing.
        response_data["errorInfo"] = "Task reported SUCCESS but its output payload could not be found in Azure."

    return response_data