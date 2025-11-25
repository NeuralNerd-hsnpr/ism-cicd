import requests
import time
import os

# Configuration
BASE_URL = "http://localhost:8000"
POLL_INTERVAL = 2

def test_sdxl_inpainting():
    print("\n--- Testing SDXL Inpainting ---")
    url = f"{BASE_URL}/inpainting"
    
    # Use dummy image URLs (replace with valid ones if needed for real testing)
    # Use dummy image URLs (replace with valid ones if needed for real testing)
    payload = {
        "parentID": "test_sdxl_parent",
        "method": "sync",
        "output": "default",
        "images": [
            {
                "imageID": "test_img_sdxl",
                "url": "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg",
                "operations": [
                    {
                        "maskUrl": "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg",
                        "phrase": "A fantasy landscape",
                        "model": "sdxl" # Note: InpaintingOperation might not have 'model' field yet, checking schemas.py
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Success!")
            print(response.json())
        else:
            print(f"Failed with status {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")

def test_pix2pix_styling():
    print("\n--- Testing Pix2Pix Styling (via Treatment) ---")
    url = f"{BASE_URL}/treatment"
    
    payload = {
        "images": [
            {
                "image_id": "test_img_1",
                "base_url": "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg",
                "operations": [
                    {
                        "type": "STYLE",
                        "style_description": "Make it look like a Van Gogh painting",
                        "model": "pix2pix" # Requesting Pix2Pix
                    }
                ]
            }
        ],
        "output": "default",
        "method": "sync"
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Success!")
            print(response.json())
        else:
            print(f"Failed with status {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")

def test_gaze_integration():
    print("\n--- Testing Gaze Integration (via Treatment) ---")
    url = f"{BASE_URL}/treatment"
    
    payload = {
        "images": [
            {
                "image_id": "test_img_2",
                "base_url": "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg", # Needs a face image ideally
                "operations": [
                    {
                        "type": "PLACE",
                        "treatment": "cap", # Triggers gaze detection
                        "location": "top",
                        "source_detected_code": "face", # Dummy detection
                        "model": "flux"
                    }
                ]
            }
        ],
        "output": "default",
        "method": "sync"
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Success!")
            print(response.json())
        else:
            print(f"Failed with status {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_sdxl_inpainting()
    test_pix2pix_styling()
    test_gaze_integration()
