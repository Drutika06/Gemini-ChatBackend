"""
Example 1: NO RAG — LLM Only (Baseline)
========================================
CONCEPT: Ask the LLM a domain-specific question WITHOUT giving it any context.
         The model can only use its training data, so it may hallucinate or
         give generic answers about topics it wasn't trained on.

WHY THIS MATTERS: This is the "before" picture. By comparing this output to
                  later RAG examples, you'll see exactly what retrieval adds.

RUN:  python3 examples/01_no_rag_baseline.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(override=True)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.0-flash")

print("=" * 60)
print("EXAMPLE 1: LLM Only — No RAG (Baseline)")
print("=" * 60)
print("Ask questions about the MBSR Handbook.")
print("The LLM has NO access to the document — watch it struggle.\n")
print("Type 'quit' to exit.\n")

while True:
    try:
        q = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!"); break
    if not q:
        continue
    if q.lower() in ("quit", "exit", "q"):
        break

    response = model.generate_content(q)
    print(f"\nAI (no context): {response.text}\n")
