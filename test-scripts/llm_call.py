import sys
import os
from pathlib import Path

# Add the main-app directory to sys.path so we can import services
# Assuming this script is in /test-scripts/ and main-app is in /main-app/
current_dir = Path(__file__).resolve().parent
main_app_dir = current_dir.parent / "main-app"
sys.path.append(str(main_app_dir))

try:
    from main_app.services import llm_service
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import llm_service. Ensure '{main_app_dir}' exists and contains 'services/llm_service.py'.")
    print(f"Error details: {e}")
    sys.exit(1)

def test_llm_service():
    print("\n--- Testing LLM Service Configuration ---")
    
    # 1. Check Environment Variables
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    print(f"GROQ_API_KEY present: {'Yes' if groq_key else 'No'}")
    print(f"OPENAI_API_KEY present: {'Yes' if openai_key else 'No'}")
    
    if not groq_key and not openai_key:
        print("ERROR: No API keys found! The LLM service requires at least one key.")
        return

    # 2. Test Object Extraction
    print("\n--- Testing extract_main_object_name ---")
    phrase = "a red lightsaber"
    print(f"Input phrase: '{phrase}'")
    try:
        result = llm_service.extract_main_object_name(phrase)
        print(f"Result: '{result}'")
    except Exception as e:
        print(f"FAILED: {e}")

    # 3. Test Dimension Estimation
    print("\n--- Testing estimate_object_dimensions ---")
    object_name = "lightsaber"
    print(f"Input object: '{object_name}'")
    try:
        dims = llm_service.estimate_object_dimensions(object_name)
        print(f"Result: {dims}")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_llm_service()
