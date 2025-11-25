# app/schemas.py

from pydantic import BaseModel, Field, HttpUrl, conint, confloat
from enum import Enum
from typing import List, Optional, Any, Union, Tuple, Literal, Annotated

# ===================================================================
# 1. ENUMS AND SHARED SUB-MODELS
# ===================================================================

class ProcessingMethod(str, Enum):
    ASYNC = "async"
    SYNC = "sync"

class OutputVerbosity(str, Enum):
    DEFAULT = "default"
    VERBOSE = "verbose"

class OperationType(str, Enum):
    STYLE = "STYLE"
    PLACE = "PLACE"
    REPLACE = "REPLACE"
    ADJUST = "ADJUST"
    REMOVE = "REMOVE"
    FILTER = "FILTER"

class PlacementType(str, Enum):
    NEAR = "NEAR"
    BEHIND = "BEHIND"
    WITHIN = "WITHIN"

class FilterType(str, Enum):
    SKETCH = "Sketch"
    CARTOON = "Cartoon"
    # ... add other available filter names here

class ModelParameters(BaseModel):
    inference_steps: Optional[conint(ge=1, le=100)] = Field(25, alias='inferenceSteps')
    guidance_scale: Optional[confloat(ge=0, le=50)] = Field(7.5, alias='guidanceScale')
    randomized_seed: Optional[bool] = Field(True, alias='randomizedSeed')
    seed: Optional[int] = 0

# ===================================================================
# 2. INDIVIDUAL OPERATION MODELS
# ===================================================================

# Base model that all operations inherit from, defining the discriminator
class BaseOperation(BaseModel):
    type: OperationType
    model: Optional[str] = None # Allow clients to specify 'flux', 'sdxl', 'pix2pix', etc.

class StyleOperation(BaseOperation):
    type: Literal[OperationType.STYLE]
    style_description: str = Field(..., alias='styleDescription')

class FilterOperation(BaseOperation):
    type: Literal[OperationType.FILTER]
    filter_type: FilterType = Field(..., alias='filterType')

class RemoveOperation(BaseOperation):
    type: Literal[OperationType.REMOVE]
    source_detected_code: str = Field(..., alias='sourceDetectedCode')
    source_detected_mask_url: Optional[HttpUrl] = Field(None, alias='sourceDetectedMaskUrl')

# A common base for ADJUST, REPLACE, and PLACE which share many fields
class TreatmentOperationBase(BaseOperation):
    source_detected_code: str = Field(..., alias='sourceDetectedCode')
    source_detected_mask_url: Optional[HttpUrl] = Field(None, alias='sourceDetectedMaskUrl')
    avoidance_mask_urls: Optional[List[HttpUrl]] = Field(None, alias='avoidanceMaskUrls')
    treatment: str
    model_parameters: Optional[ModelParameters] = Field(default_factory=ModelParameters, alias='modelParameters')

class AdjustOperation(TreatmentOperationBase):
    type: Literal[OperationType.ADJUST]

class ReplaceOperation(TreatmentOperationBase):
    type: Literal[OperationType.REPLACE]

class PlaceOperation(TreatmentOperationBase):
    type: Literal[OperationType.PLACE]
    placement_type: PlacementType = Field(..., alias='placementType')
    placement_rotation: Optional[int] = None
    location: str
    detected_object_pixels_per_cm: Optional[int] = Field(None, alias='detectedObjectPixelsPerCm')
    treatment_target_size: Optional[Tuple[int, int]] = Field(None, alias='treatmentTargetSize')

# The Discriminated Union: Pydantic will automatically use the correct model
# from this list based on the value of the 'type' field.
AnyTreatmentOperation = Annotated[
    Union[
        StyleOperation,
        FilterOperation,
        RemoveOperation,
        AdjustOperation,
        ReplaceOperation,
        PlaceOperation,
    ],
    Field(discriminator='type')
]

# ===================================================================
# 3. TOP-LEVEL REQUEST/RESPONSE MODELS
# ===================================================================

# --- INPAINTING MODELS (Unchanged) ---
class InpaintingOperation(BaseModel):
    mask_url: HttpUrl = Field(..., alias='maskUrl')
    phrase: str
    seed: Optional[int] = None
    guidance_scale: Optional[float] = Field(None, alias='guidanceScale')
    num_inference_steps: Optional[int] = Field(None, alias='numInferenceSteps')
    model: Optional[str] = None

class InpaintingImageTask(BaseModel):
    image_id: str = Field(..., alias='imageID')
    base_url: HttpUrl = Field(..., alias='url')
    operations: List[InpaintingOperation]

class ComplexInpaintingRequest(BaseModel):
    parent_id: str = Field(..., alias='parentID')
    method: ProcessingMethod = ProcessingMethod.ASYNC
    output: OutputVerbosity = OutputVerbosity.DEFAULT
    images: List[InpaintingImageTask]


# --- TREATMENT MODELS (New) ---
class TreatmentImageTask(BaseModel):
    image_id: str = Field(..., alias='imageID')
    base_url: HttpUrl = Field(..., alias='url')
    operations: List[AnyTreatmentOperation]

class ComplexTreatmentRequest(BaseModel):
    parent_id: str = Field(..., alias='parentID')
    method: ProcessingMethod = ProcessingMethod.ASYNC
    output: OutputVerbosity = OutputVerbosity.DEFAULT
    images: List[TreatmentImageTask]

# Common Union for Validator
ImageTask = Union[InpaintingImageTask, TreatmentImageTask]


# --- BACKGROUND REMOVAL MODELS (New) ---
class BackgroundRemovalImageTask(BaseModel):
    image_id: str = Field(..., alias='imageID')
    base_url: HttpUrl = Field(..., alias='url')

class BackgroundRemovalRequest(BaseModel):
    parent_id: str = Field(..., alias='parentID')
    method: ProcessingMethod = ProcessingMethod.ASYNC
    output: OutputVerbosity = OutputVerbosity.DEFAULT
    images: List[BackgroundRemovalImageTask]


# --- BACKGROUND INSERTION MODELS (New) ---
class BackgroundInsertionImageTask(BaseModel):
    image_id: str = Field(..., alias='imageID')
    base_url: HttpUrl = Field(..., alias='url')
    insertion_url: HttpUrl = Field(..., alias='insertionUrl')

class BackgroundInsertionRequest(BaseModel):
    parent_id: str = Field(..., alias='parentID')
    method: ProcessingMethod = ProcessingMethod.ASYNC
    output: OutputVerbosity = OutputVerbosity.DEFAULT
    images: List[BackgroundInsertionImageTask]


# --- BACKGROUND REPLACEMENT MODELS (New) ---
class BackgroundReplacementImageTask(BaseModel):
    image_id: str = Field(..., alias='imageID')
    base_url: HttpUrl = Field(..., alias='url')
    insertion_url: HttpUrl = Field(..., alias='insertionUrl')

class BackgroundReplacementRequest(BaseModel):
    parent_id: str = Field(..., alias='parentID')
    method: ProcessingMethod = ProcessingMethod.ASYNC
    output: OutputVerbosity = OutputVerbosity.DEFAULT
    images: List[BackgroundReplacementImageTask]


# --- VIDEO GENERATION MODELS (New) ---
class VideoGenerationImageTask(BaseModel):
    image_id: str = Field(..., alias='imageID')
    base_url: Optional[HttpUrl] = Field(None, alias='url') # Optional for text-to-video
    prompt: str
    negative_prompt: Optional[str] = Field("", alias='negativePrompt')
    seed: Optional[int] = 0
    num_frames: Optional[int] = Field(121, alias='numFrames')

class VideoGenerationRequest(BaseModel):
    parent_id: str = Field(..., alias='parentID')
    method: ProcessingMethod = ProcessingMethod.ASYNC
    output: OutputVerbosity = OutputVerbosity.DEFAULT
    images: List[VideoGenerationImageTask]


# --- GENERIC RESPONSE MODELS (Unchanged) ---
class TaskCreationResponse(BaseModel):
    task_id: str


# --- Final Result and Summary Schemas ---

class ProcessingStep(BaseModel):
    """Represents a detailed step within the AI/image processing pipeline."""
    step: int
    action: str
    ai_models: Optional[str] = Field(None, alias='aiModels')
    methods: Optional[str] = None
    version: Optional[float] = None
    elapsed_seconds: float = Field(..., alias='elapsedSeconds')

class ImageResult(BaseModel):
    """Represents the final result for a single processed image."""
    image_id: str = Field(..., alias='imageID')
    success: bool
    error: Optional[str] = None
    image_temporary: Optional[HttpUrl] = Field(None, alias='imageTemporary')
    image_permanent: Optional[HttpUrl] = Field(None, alias='imagePermanent')
    steps: List[ProcessingStep]

class OverallStep(BaseModel):
    """Represents a high-level step in the overall orchestration process."""
    step: int
    action: str
    elapsed_seconds: float = Field(..., alias='elapsedSeconds')

class OutputData(BaseModel):
    """The main 'output' block containing all results and metadata."""
    parent_id: str = Field(..., alias='parentID')
    elapsed_seconds: float = Field(..., alias='elapsedSeconds')
    steps: List[OverallStep]
    results: List[ImageResult]

class ComplexInpaintingResponse(BaseModel):
    """The root model for a successful response payload (used for sync requests)."""
    success: bool
    output: OutputData

# Aliases for other operation responses (they share the same structure)
BackgroundRemovalResponse = ComplexInpaintingResponse
BackgroundInsertionResponse = ComplexInpaintingResponse
BackgroundReplacementResponse = ComplexInpaintingResponse
VideoGenerationResponse = ComplexInpaintingResponse
ComplexTreatmentResponse = ComplexInpaintingResponse

class TaskFailureResponse(BaseModel):
    """Standard response when a task has failed (used for sync requests)."""
    task_id: str = Field(..., alias='taskId')
    status: str = Field("FAILURE", description="The final status of the task.")
    error_info: str = Field(..., alias='errorInfo', description="The error message or traceback that caused the failure.")

class TaskSummaryResponse(BaseModel):
    """
    Provides a comprehensive summary of a task's status and its associated data.
    """
    task_id: str = Field(..., alias='taskId')
    status: str
    input_payload: Optional[dict] = Field(None, alias='inputPayload')
    output_payload: Optional[dict] = Field(None, alias='outputPayload')
    error_info: Optional[Any] = Field(None, alias='errorInfo')
