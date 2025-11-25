# test_sdxl_generation_api.py

import httpx
import os

# --- Configuration ---
SDXL_API_URL = "http://localhost:8002/generate"
OUTPUT_IMAGE_PATH = "test_images/sdxl_generation_output.png"

def run_sdxl_generation_api_test():
    """
    Sends a test request to the running SDXL Image Generation service.
    """
    print("--- Starting SDXL Image Generation API Test ---")

    try:
        # Prepare the multipart/form-data payload.
        # Since there are no files, we just use the 'data' parameter.
        data_payload = {
            'prompt': 'A photorealistic portrait of an astronaut on Mars, detailed, cinematic lighting, 8k',
            'negative_prompt': 'cartoon, drawing, painting, blurry, low quality',
            'seed': 42
        }

        print(f"Sending request to: {SDXL_API_URL}")
        print(f"Payload data: {data_payload}")
        
        # This will take a long time, so a long timeout is essential.
        with httpx.Client(timeout=600.0) as client:
            response = client.post(
                SDXL_API_URL,
                data=data_payload
            )
            response.raise_for_status()

        print(f"Success! Received {len(response.content)} bytes of image data.")

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(OUTPUT_IMAGE_PATH), exist_ok=True)
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
        print(f"Error: Could not connect to the API at {SDXL_API_URL}.")
    except Exception as e:
        print(f"\n--- TEST FAILED ---")
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_sdxl_generation_api_test()