import httpx
import os

# --- Configuration ---
BG_REMOVAL_API_URL = "http://localhost:8003/remove_background"
BASE_IMAGE_PATH = "test_images/base_image.jpg" # An image with a clear foreground subject
OUTPUT_IMAGE_PATH = "test_images/bg_removal_output.png"

def run_bg_removal_api_test():
    """
    Sends a test request to the running Background Removal service.
    """
    print("--- Starting Background Removal API Test ---")

    if not os.path.exists(BASE_IMAGE_PATH):
        print(f"ERROR: Base image not found at '{BASE_IMAGE_PATH}'")
        return

    print(f"Loading base image from: {BASE_IMAGE_PATH}")

    try:
        with open(BASE_IMAGE_PATH, "rb") as image_f:
            files_payload = {
                'image': ('image.png', image_f, 'image/png')
            }

            print(f"Sending request to: {BG_REMOVAL_API_URL}")

            with httpx.Client(timeout=300.0) as client:
                response = client.post(BG_REMOVAL_API_URL, files=files_payload)
                response.raise_for_status()

            print(f"Success! Received {len(response.content)} bytes of image data.")

            with open(OUTPUT_IMAGE_PATH, "wb") as output_f:
                output_f.write(response.content)
            
            print(f"Output image saved to: {OUTPUT_IMAGE_PATH}")
            print("NOTE: Open the output file to verify the background is transparent.")

    except httpx.HTTPStatusError as e:
        print(f"\n--- TEST FAILED ---")
        print(f"Error: The API returned a status code {e.response.status_code}")
        try:
            print(f"API Response: {e.response.json()}")
        except:
            print(f"API Response (raw): {e.response.text}")
    except httpx.RequestError as e:
        print(f"\n--- TEST FAILED ---")
        print(f"Error: Could not connect to the API at {BG_REMOVAL_API_URL}.")
        print("Please ensure the Background Removal API service is running.")
    except Exception as e:
        print(f"\n--- TEST FAILED ---")
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_bg_removal_api_test()