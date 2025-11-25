import httpx
import time
import json

# Configuration
API_URL = "http://localhost:8000/remove-background"
IMAGE_URL = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg" # Example image

def test_background_removal_sync():
    print("\n--- Testing Synchronous Background Removal ---")
    payload = {
        "parentID": "test-sync-123",
        "method": "sync",
        "output": "default",
        "images": [
            {
                "imageID": "img1",
                "url": IMAGE_URL
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

def test_background_removal_async():
    print("\n--- Testing Asynchronous Background Removal ---")
    payload = {
        "parentID": "test-async-123",
        "method": "async",
        "output": "default",
        "images": [
            {
                "imageID": "img2",
                "url": IMAGE_URL
            }
        ]
    }
    
    start_time = time.time()
    try:
        response = httpx.post(API_URL, json=payload, timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 202:
            data = response.json()
            task_id = data.get("task_id")
            print(f"Task Submitted. Task ID: {task_id}")
            
            # Poll for result
            result_url = f"http://localhost:8000/results/{task_id}"
            for _ in range(30): # Poll for 30 seconds
                time.sleep(1)
                res_response = httpx.get(result_url)
                if res_response.status_code == 200:
                    print("Task Completed!")
                    print(json.dumps(res_response.json(), indent=2))
                    break
                elif res_response.status_code != 202:
                    print(f"Task Failed or Error: {res_response.status_code}")
                    print(res_response.text)
                    break
                print(".", end="", flush=True)
            else:
                print("\nTimeout waiting for async task.")

        else:
            print("Failed to submit task!")
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")
    print(f"\nElapsed Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    test_background_removal_sync()
    test_background_removal_async()
