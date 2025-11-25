# test_sd_style_api.py

import httpx
import os

# --- Configuration ---
# The URL of your locally running SD Styling service.
# Make sure the port matches the one in your uvicorn command (e.g., 8006).
SD_STYLE_API_URL = "http://localhost:8005/style"

# The path to your local test image.
BASE_IMAGE_PATH = "test_images/base_image.jpg"

# We will save the output for each model to a separate file for comparison.
OUTPUT_IMAGE_PATH_PIX2PIX = "test_images/sd_pix2pix_output.png"
OUTPUT_IMAGE_PATH_CONTROLNET = "test_images/sd_controlnet_output.png"


def test_single_model(model_name: str, prompt: str, output_path: str):
    """
    A helper function to test one specific model in the style service.
    """
    print(f"\n--- Testing Model: '{model_name}' ---")

    if not os.path.exists(BASE_IMAGE_PATH):
        print(f"ERROR: Base image not found at '{BASE_IMAGE_PATH}'")
        return

    print(f"Loading base image from: {BASE_IMAGE_PATH}")

    try:
        with open(BASE_IMAGE_PATH, "rb") as image_f:
            
            # 1. Prepare the multipart/form-data payload
            files_payload = {
                'image': ('image.png', image_f, 'image/png')
            }
            
            data_payload = {
                'prompt': prompt,
                'model': model_name,
                'num_inference_steps': 50 # Example of an optional parameter
            }
            
            # The 'pix2pix' model has a unique parameter, let's test it.
            if model_name == "pix2pix":
                data_payload['image_guidance_scale'] = 1.5

            print(f"Sending request to: {SD_STYLE_API_URL}")
            print(f"Payload data: {data_payload}")

            # 2. Make the request with a long timeout suitable for diffusion models.
            # A 10-minute timeout is a safe starting point.
            with httpx.Client(timeout=600.0) as client:
                response = client.post(
                    SD_STYLE_API_URL,
                    files=files_payload,
                    data=data_payload
                )
                response.raise_for_status()

            # 3. Handle the successful response.
            print(f"Success! Received {len(response.content)} bytes of image data.")
            with open(output_path, "wb") as output_f:
                output_f.write(response.content)
            print(f"Output image saved to: {output_path}")

    except httpx.HTTPStatusError as e:
        print(f"\n--- TEST FAILED for model '{model_name}' ---")
        print(f"Error: The API returned a status code {e.response.status_code}")
        try:
            print(f"API Response: {e.response.json()}")
        except:
            print(f"API Response (raw): {e.response.text}")

    except httpx.RequestError as e:
        print(f"\n--- TEST FAILED for model '{model_name}' ---")
        print(f"Error: Could not connect to the API at {SD_STYLE_API_URL}.")
        print("Please ensure the SD Style API service is running and accessible.")
    
    except Exception as e:
        print(f"\n--- TEST FAILED for model '{model_name}' ---")
        print(f"An unexpected error occurred: {e}")


def run_all_style_api_tests():
    """
    Runs a series of tests against the SD Style API, one for each model.
    """
    print("--- Starting Stable Diffusion Style API Test Suite ---")
    
    # Test Case 1: Use the 'pix2pix' model
    test_single_model(
        model_name="pix2pix",
        prompt="turn the person into a bronze statue",
        output_path=OUTPUT_IMAGE_PATH_PIX2PIX
    )
    
    # Test Case 2: Use the 'controlnet' model
    test_single_model(
        model_name="controlnet",
        prompt="make the image look like a watercolor painting",
        output_path=OUTPUT_IMAGE_PATH_CONTROLNET
    )
    
    print("\n--- Test Suite Finished ---")


if __name__ == "__main__":
    run_all_style_api_tests()