"""
Quick test script to validate Gemini API key
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found in .env")
    exit(1)

print(f"Testing API key: {api_key[:10]}...{api_key[-5:]}\n")

try:
    genai.configure(api_key=api_key)
    
    # Test 1: Check if embedding works
    print("Testing embedding model...")
    result = genai.embed_content(
        model="models/text-embedding-004",
        content="test",
        task_type="retrieval_document"
    )
    print("✅ Embedding API works")
    
    # Test 2: Check if generation works
    print("Testing generative model...")
    model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")
    response = model.generate_content("Say 'API key is valid'")
    print("✅ Generation API works")
    print(f"   Response: {response.text[:50]}...\n")
    
    print("🎉 All tests passed! Your API key is valid.")
    
except Exception as e:
    print(f"❌ API Error: {str(e)}\n")
    print("Possible causes:")
    print("  1. API key expired - regenerate at https://aistudio.google.com/app/apikeys")
    print("  2. API not enabled - enable Generative Language API in Google Cloud Console")
    print("  3. Quota exceeded - check usage in Google Cloud Console")
    exit(1)
