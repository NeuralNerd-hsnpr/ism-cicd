# services/llm_service.py

import os
import ast
from typing import List, Dict
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

# --- 1. CONFIGURATION AND INITIALIZATION ---
# Load environment variables once when the module is imported.
load_dotenv()

# Define a custom exception for this service for clear error handling.
class LLMServiceError(Exception):
    """Custom exception for when all LLM providers fail."""
    pass

# Initialize clients at the module level.
# This is efficient as it happens only once when the Celery worker starts.
groq_client = None
if os.getenv("GROQ_API_KEY"):
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

openai_client = None
if os.getenv("OPENAI_API_KEY"):
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
# Define the provider fallback chain as a module-level constant.
# This makes the logic clear and easy to reconfigure.
_PROVIDERS = [
    {
        "name": "Groq",
        "client": groq_client,
        "models": ["llama3-70b-8192"], # Updated model name
    },
    {
        "name": "OpenAI",
        "client": openai_client,
        "models": ["gpt-4o-mini", "gpt-4o"],
    },
]

# --- 2. CORE PRIVATE FUNCTION ---

def _get_completion(messages: List[Dict[str, str]]) -> str:
    """
    The main workhorse function that attempts to get a completion from a chain of providers.
    It tries each provider and its models in order until one succeeds.
    """
    last_exception = None
    for provider in _PROVIDERS:
        if not provider["client"]:
            print(f"Skipping provider '{provider['name']}' as its client is not configured.")
            continue

        for model in provider["models"]:
            try:
                print(f"Attempting to call provider '{provider['name']}' with model '{model}'...")
                response = provider["client"].chat.completions.create(
                    model=model, messages=messages
                )
                if hasattr(response, "usage") and response.usage:
                    print(f"[{provider['name']}] Success! Tokens used: {response.usage.total_tokens}")
                # On success, strip whitespace and return the content immediately.
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Provider '{provider['name']}' with model '{model}' failed: {e}")
                last_exception = e
                # Continue to the next model in the list.
                continue
    
    # If both loops complete without returning, all providers and models have failed.
    raise LLMServiceError(f"All LLM providers failed to respond. Last error: {last_exception}")


# --- 3. PUBLIC BUSINESS LOGIC FUNCTIONS ---

def extract_main_object_name(phrase: str) -> str:
    """
    Given a descriptive phrase, extracts the main object as a single, lowercase word.
    """
    system_prompt = "Your response must contain only one word in lowercase. If the object is a compound noun like 'light saber', return it as 'lightsaber'."
    user_prompt = f"From the following statement, extract only the main object: {phrase}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    return _get_completion(messages)

def estimate_object_dimensions(phrase: str) -> dict:
    """
    Given an object name, returns its approximate real-world size in cm as a dictionary.
    """
    system_prompt = (
        "You are an assistant that only returns a Python dictionary containing 'actual_width' and "
        "'actual_height' in centimeters. Do not include any other text, formatting, or markdown. "
        "Your output must be parsable by Python's ast.literal_eval()."
    )
    user_prompt = f"""
    Return the approximate real-world size for the object: {phrase}.
    Example for 'light saber': {{'actual_width': 5, 'actual_height': 100}}
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response_str = _get_completion(messages)
    
    try:
        # Safely evaluate the string response to a Python dictionary.
        dimensions = ast.literal_eval(response_str)
        if not isinstance(dimensions, dict) or 'actual_width' not in dimensions or 'actual_height' not in dimensions:
            raise ValueError("LLM response is not a valid dimension dictionary.")
        return dimensions
    except (ValueError, SyntaxError) as e:
        print(f"Failed to parse LLM response for dimensions: '{response_str}'. Error: {e}")
        raise LLMServiceError(f"LLM returned a response that could not be parsed into a dictionary: {response_str}") from e