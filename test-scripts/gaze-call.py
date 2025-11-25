# test_gaze_api.py

import httpx
import os

# --- Configuration ---
GAZE_API_URL = "http://localhost:8002/get_direction_by_mask"
BASE_IMAGE_PATH = "test_images/base_image_with_face.png" # An image with a face
FACE_MASK_PATH = "test_images/base_face_mask.png" # A mask isolating that face
OUTPUT_FILE = "test_images/gaze_result.txt"

def run_gaze_api_test():
    """
    Sends a test request to the running Gaze Estimation service.
    """
    print("--- Starting Gaze Estimation API Test ---")

    if not all(os.path.exists(p) for p in [BASE_IMAGE_PATH, FACE_MASK_PATH]):
        print(f"ERROR: Missing test files. Ensure '{BASE_IMAGE_PATH}' and '{FACE_MASK_PATH}' exist.")
        return

    try:
        with open(BASE_IMAGE_PATH, "rb") as base_f, open(FACE_MASK_PATH, "rb") as mask_f:
            files_payload = {
                'base_image': ('base_image.png', base_f, 'image/png'),
                'face_mask': ('face_mask.png', mask_f, 'image/png')
            }

            print(f"Sending request to: {GAZE_API_URL}")

            with httpx.Client(timeout=60.0) as client:
                response = client.post(GAZE_API_URL, files=files_payload)
                response.raise_for_status()

            result = response.json()
            direction = result.get("direction")
            
            print(f"\n--- TEST SUCCEEDED ---")
            print(f"API Response: {result}")
            print(f"Detected Gaze Direction: {direction}")

            with open(OUTPUT_FILE, "w") as f:
                f.write(f"Detected Gaze Direction: {direction}\n")
            print(f"Result saved to: {OUTPUT_FILE}")

    except httpx.HTTPStatusError as e:
        print(f"\n--- TEST FAILED ---")
        print(f"Error: The API returned a status code {e.response.status_code}")
        try:
            print(f"API Response: {e.response.json()}")
        except:
            print(f"API Response (raw): {e.response.text}")
    except httpx.RequestError as e:
        print(f"\n--- TEST FAILED ---")
        print(f"Error: Could not connect to the API at {GAZE_API_URL}.")
        print("Please ensure the Gaze Estimation API service is running.")
    except Exception as e:
        print(f"\n--- TEST FAILED ---")
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_gaze_api_test()