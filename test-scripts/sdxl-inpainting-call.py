# test_sd_inpainting_api.py

import httpx
import os

# --- Configuration ---
SD_INPAINT_API_URL = "http://localhost:8006/inpaint"
BASE_IMAGE_PATH = "test_images/base_image.jpg"
MASK_IMAGE_PATH = "test_images/base_mask.png"
OUTPUT_IMAGE_PATH = "test_images/sdxl_inpainting_output.png"

def run_sd_inpainting_api_test():
    """
    Sends a test request to the running Stable Diffusion Inpainting service.
    """
    print("--- Starting Stable Diffusion Inpainting API Test ---")

    if not all(os.path.exists(p) for p in [BASE_IMAGE_PATH, MASK_IMAGE_PATH]):
        print(f"ERROR: Missing test files. Ensure '{BASE_IMAGE_PATH}' and '{MASK_IMAGE_PATH}' exist.")
        return

    try:
        with open(BASE_IMAGE_PATH, "rb") as base_f, open(MASK_IMAGE_PATH, "rb") as mask_f:
            files_payload = {
                'base_image': ('base_image.png', base_f, 'image/png'),
                'mask_image': ('mask_image.png', mask_f, 'image/png')
            }

            data_payload = {
                'prompt': 'a beautiful oil painting of a majestic mountain',
                'strength': 0.9,
                'seed': 12345
            }

            print(f"Sending request to: {SD_INPAINT_API_URL}")
            print(f"Payload data: {data_payload}")

            with httpx.Client(timeout=300.0) as client:
                response = client.post(
                    SD_INPAINT_API_URL,
                    files=files_payload,
                    data=data_payload
                )
                response.raise_for_status()

            print(f"Success! Received {len(response.content)} bytes of image data.")

            with open(OUTPUT_IMAGE_PATH, "wb") as output_f:
                output_f.write(response.content)
            
            print(f"Output image saved to: {OUTPUT_IMAGE_PATH}")

    except httpx.HTTPStatusError as e:
        print(f"\n--- TEST FAILED ---")
        print(f"Error: The API returned a status code {e.response.status_code}")
        try: print(f"API Response: {e.response.json()}")
        except: print(f"API Response (raw): {e.response.text}")
    except httpx.RequestError as e:
        print(f"\n--- TEST FAILED ---")
        print(f"Error: Could not connect to the API at {SD_INPAINT_API_URL}.")
    except Exception as e:
        print(f"\n--- TEST FAILED ---")
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_sd_inpainting_api_test()