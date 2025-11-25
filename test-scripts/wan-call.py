import requests
import sys

WAN_API_URL = "http://localhost:8007/generate_video"

def test_wan_service():
    print(f"Testing Wan Service at {WAN_API_URL}...")
    
    prompt = "A cinematic drone shot of a futuristic city at sunset"
    negative_prompt = "blurry, low quality"
    
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt
    }
    
    try:
        # Test Text-to-Video
        print("Sending Text-to-Video request...")
        response = requests.post(WAN_API_URL, data=payload, timeout=600)
        
        if response.status_code == 200:
            print("Success! Video generated.")
            with open("wan_test_output.mp4", "wb") as f:
                f.write(response.content)
            print("Saved to wan_test_output.mp4")
        else:
            print(f"Failed with status {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_wan_service()
