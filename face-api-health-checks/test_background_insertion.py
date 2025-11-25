import httpx
import time
import json

# Configuration
API_URL = "http://localhost:8000/insert-background"
# Transparent image (Foreground)
FOREGROUND_URL = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg" # Placeholder: Ideally this should be a transparent PNG
# Background image
BACKGROUND_URL = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg" # Placeholder

def test_background_insertion_sync():
    print("\n--- Testing Synchronous Background Insertion ---")
    payload = {
        "parentID": "test-insert-sync-123",
        "method": "sync",
        "output": "default",
        "images": [
            {
                "imageID": "img1",
                "url": FOREGROUND_URL,
                "insertionUrl": BACKGROUND_URL
            }
        ]
    }
    
    start_time = time.time()
    try:
        response = httpx.post(API_URL, json=payload, timeout=300)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Success!")
            print(json.dumps(response.json(), indent=2))
        else:
            print("Failed!")
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")
    print(f"Elapsed Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    test_background_insertion_sync()
