"""
List available models for embeddings and text generation
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("Available models:\n")

try:
    for model in genai.list_models():
        print(f"Name: {model.name}")
        print(f"Display: {model.display_name}")
        print(f"Supported methods: {model.supported_generation_methods}")
        print()
except Exception as e:
    print(f"Error: {e}")
