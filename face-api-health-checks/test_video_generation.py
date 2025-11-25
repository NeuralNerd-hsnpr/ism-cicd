import requests
import json
import time
import sys

API_URL = "http://localhost:8000/generate-video"

def test_video_generation():
    print(f"Testing Main App Video Generation at {API_URL}...")
    
    payload = {
        "parentID": "test-video-parent-1",
        "method": "async",
        "output": "default",
        "images": [
            {
                "imageID": "img1",
                "prompt": "A cute robot dancing in the rain",
                "negativePrompt": "bad quality",
                "numFrames": 81
            }
        ]
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            task_id = response.json().get("task_id")
            print(f"Task created successfully. Task ID: {task_id}")
            print("Check Celery logs for progress and Azure storage for output.")
        else:
            print(f"Failed with status {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_video_generation()
