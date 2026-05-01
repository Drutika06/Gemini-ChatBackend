"""
List Available Embedding Models
================================
Simple script to list all supported embedding models from Google Generative AI API.

RUN:  python3 list_embedding_models.py
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(override=True)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("=" * 70)
print("SUPPORTED EMBEDDING MODELS")
print("=" * 70)

try:
    models = genai.list_models()
    embedding_models = [m for m in models if "embed" in m.name.lower()]
    
    if embedding_models:
        print(f"\nFound {len(embedding_models)} embedding model(s):\n")
        for i, model in enumerate(embedding_models, 1):
            print(f"{i}. {model.name}")
            print(f"   Display Name: {model.display_name}")
            print(f"   Description: {model.description}")
            print()
    else:
        print("\nNo embedding models found.")
        print("\nAll available models:")
        for model in models:
            print(f"  - {model.name}")

except Exception as e:
    print(f"\nError: {e}")
    print("\nMake sure:")
    print("  1. GEMINI_API_KEY is set in .env")
    print("  2. You have internet connectivity")
