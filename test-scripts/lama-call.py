# test_lama_api.py

import httpx
import os

# --- Configuration ---
# The URL of your locally running Lama Removal service.
# Make sure the port matches the one you use to run uvicorn (e.g., 8005).
LAMA_API_URL = "http://localhost:8002/remove"

# The paths to your local test images.
BASE_IMAGE_PATH = "test_images/base_image.jpg"
MASK_IMAGE_PATH = "test_images/base_mask.png"

# The path where the output image will be saved.
OUTPUT_IMAGE_PATH = "test_images/lama_output.png"


def run_lama_api_test():
    """
    Sends a test request to the running Lama Removal service.
    """
    print("--- Starting Lama Removal API Test ---")

    # 1. Verify that the input files exist.
    if not os.path.exists(BASE_IMAGE_PATH):
        print(f"ERROR: Base image not found at '{BASE_IMAGE_PATH}'")
        return
    if not os.path.exists(MASK_IMAGE_PATH):
        print(f"ERROR: Mask image not found at '{MASK_IMAGE_PATH}'")
        return

    print(f"Loading base image from: {BASE_IMAGE_PATH}")
    print(f"Loading mask image from: {MASK_IMAGE_PATH}")

    # 2. Prepare the multipart/form-data payload.
    # The Lama service expects the files to be named 'image' and 'mask'.
    try:
        with open(BASE_IMAGE_PATH, "rb") as image_f, open(MASK_IMAGE_PATH, "rb") as mask_f:
            
            files_payload = {
                'image': ('image.png', image_f, 'image/png'),
                'mask': ('mask.png', mask_f, 'image/png')
            }

            print(f"Sending request to: {LAMA_API_URL}")

            # 3. Make the request using httpx.
            with httpx.Client(timeout=300.0) as client:
                response = client.post(
                    LAMA_API_URL,
                    files=files_payload
                )

                response.raise_for_status()

            # 4. Handle the successful response.
            print(f"Success! Received {len(response.content)} bytes of image data.")

            with open(OUTPUT_IMAGE_PATH, "wb") as output_f:
                output_f.write(response.content)
            
            print(f"Output image saved to: {OUTPUT_IMAGE_PATH}")

    except httpx.HTTPStatusError as e:
        print(f"\n--- TEST FAILED ---")
        print(f"Error: The API returned a status code {e.response.status_code}")
        try:
            print(f"API Response: {e.response.json()}")
        except:
            print(f"API Response (raw): {e.response.text}")

    except httpx.RequestError as e:
        print(f"\n--- TEST FAILED ---")
        print(f"Error: Could not connect to the API at {LAMA_API_URL}.")
        print("Please ensure the Lama Removal API service is running and accessible at that address.")
    
    except Exception as e:
        print(f"\n--- TEST FAILED ---")
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    run_lama_api_test()