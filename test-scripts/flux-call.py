# test_flux_api.py

import httpx
import os

# --- Configuration ---
# The URL of your locally running Flux API service.
# Make sure the port matches the one you use to run uvicorn.
FLUX_API_URL = "http://localhost:8001/inpaint" # adopt to ready to work flux API instance URL

# The paths to your local test images.
BASE_IMAGE_PATH = "test_images/base_image.jpg"
MASK_IMAGE_PATH = "test_images/base_mask.png"

# The path where the output image will be saved.
OUTPUT_IMAGE_PATH = "test_images/flux_output.png"


def run_flux_api_test():
    """
    Sends a test request to the running Flux API service.
    """
    print("--- Starting Flux API Test ---")

    # 1. Verify that the input files exist before making the request.
    if not os.path.exists(BASE_IMAGE_PATH):
        print(f"ERROR: Base image not found at '{BASE_IMAGE_PATH}'")
        return
    if not os.path.exists(MASK_IMAGE_PATH):
        print(f"ERROR: Mask image not found at '{MASK_IMAGE_PATH}'")
        return

    print(f"Loading base image from: {BASE_IMAGE_PATH}")
    print(f"Loading mask image from: {MASK_IMAGE_PATH}")

    # 2. Prepare the multipart/form-data payload.
    # We open the files in binary read mode ('rb').
    try:
        with open(BASE_IMAGE_PATH, "rb") as base_f, open(MASK_IMAGE_PATH, "rb") as mask_f:
            
            # 'files' dictionary for the image binary data
            files_payload = {
                'base_image': ('base_image.png', base_f, 'image/png'),
                'mask_image': ('mask_image.png', mask_f, 'image/png')
            }

            # 'data' dictionary for the other form fields
            data_payload = {
                'prompt': 'a photorealistic golden crown',
                'seed': 42,
                'num_inference_steps': 30
            }

            print(f"Sending request to: {FLUX_API_URL}")
            print(f"Payload data: {data_payload}")

            # 3. Make the request using httpx.
            with httpx.Client(timeout=300.0) as client:
                response = client.post(
                    FLUX_API_URL,
                    files=files_payload,
                    data=data_payload
                )

                # Check for any HTTP errors (4xx or 5xx)
                response.raise_for_status()

            # 4. Handle the successful response.
            print(f"Success! Received {len(response.content)} bytes of image data.")

            # Save the resulting image bytes to a file.
            with open(OUTPUT_IMAGE_PATH, "wb") as output_f:
                output_f.write(response.content)
            
            print(f"Output image saved to: {OUTPUT_IMAGE_PATH}")

    except httpx.HTTPStatusError as e:
        print(f"\n--- TEST FAILED ---")
        print(f"Error: The API returned a status code {e.response.status_code}")
        # Try to print the detailed error message from the API's JSON response
        try:
            print(f"API Response: {e.response.json()}")
        except:
            print(f"API Response (raw): {e.response.text}")

    except httpx.RequestError as e:
        print(f"\n--- TEST FAILED ---")
        print(f"Error: Could not connect to the API at {FLUX_API_URL}.")
        print("Please ensure the Flux API service is running and accessible at that address.")
    
    except Exception as e:
        print(f"\n--- TEST FAILED ---")
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    run_flux_api_test()