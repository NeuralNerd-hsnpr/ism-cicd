from diffusers import FluxFillPipeline
from diffusers.utils import load_image
from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel
from dotenv import load_dotenv
from huggingface_hub import login
import torch
import os
from PIL import Image


class ModelManager:
    def __init__(self):
        # These will be populated by the load_models method
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

    def load_models(self):
        """
        Loads and initializes the FLUX-Fill model and pipeline.
        This is the slow, one-time setup.
        """
        print("MODEL_MANAGER: Loading environment variables and logging into Hugging Face...")
        load_dotenv()
        login(token=os.getenv("HUGGING_FACE_API_TOKEN"))

        print("MODEL_MANAGER: Loading Nunchaku Flux Transformer...")
        transformer = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-fill-dev")

        print("MODEL_MANAGER: Loading FLUX-Fill pipeline...")
        self.pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            transformer=transformer,
            torch_dtype=self.torch_dtype
        ).to(self.device)

        print(f"MODEL_MANAGER: FLUX-Fill model loaded and moved to {self.device}.")

    def run_inference(
            self,
            prompt: str,
            base_image: Image.Image,
            mask_image: Image.Image,
            seed: int | None = None,
            # Default values can be defined here
            guidance_scale: float = 7.5,
            num_inference_steps: int = 25
    ) -> Image.Image:
        """
        Runs the inpainting inference using the pre-loaded pipeline.
        Accepts PIL Images directly.
        """
        if self.pipe is None:
            raise RuntimeError("Model pipeline has not been loaded. Call load_models() first.")

        # Set the seed for reproducibility
        if seed is None:
            seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()
        generator = torch.Generator(self.device).manual_seed(seed)

        print(f"MODEL_MANAGER: Running inference with seed {seed}...")

        # Perform inpainting using the diffusion pipeline
        output_image = self.pipe(
            prompt=prompt,
            image=base_image,
            mask_image=mask_image,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]

        print("MODEL_MANAGER: Inference complete.")
        return output_image.resize(base_image.size)


# Create the single, shared instance of the manager
model_manager = ModelManager()
