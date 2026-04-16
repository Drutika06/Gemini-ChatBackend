"""
Example 2: Simplest RAG — Keyword Search + Prompt Stuffing
===========================================================
CONCEPT: The most basic RAG possible:
  1. KNOWLEDGE BASE — A few hardcoded text paragraphs (simulating a document).
  2. RETRIEVE       — Find paragraphs containing the user's keywords.
  3. AUGMENT        — Stuff matching paragraphs into the LLM prompt.
  4. GENERATE       — LLM answers using ONLY the provided context.

NEW IDEA: "Prompt stuffing" — we literally paste retrieved text into the prompt.
          This is the core pattern behind ALL RAG systems.

RUN:  python3 examples/02_simple_keyword_rag.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from click import prompt
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(override=True)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")

# ── STEP 1: Our "knowledge base" — just a list of strings ──
KNOWLEDGE_BASE = [
    "MBSR stands for Mindfulness-Based Stress Reduction. It was developed by Jon Kabat-Zinn at the University of Massachusetts Medical Center in 1979.",
    "The MBSR program is typically 8 weeks long and includes weekly group sessions of 2.5 hours each, plus a full-day retreat in week 6.",
    "Core MBSR practices include body scan meditation, sitting meditation, gentle yoga, and walking meditation.",
    "The body scan is a 45-minute practice where participants systematically bring attention to each part of the body from toes to head.",
    "Mindful eating is often introduced in session 1. Participants eat a single raisin very slowly, noticing its texture, smell, taste, and the act of chewing.",
    "Research shows MBSR can reduce symptoms of anxiety, depression, chronic pain, and improve immune function.",
    "The 'stress reaction cycle' describes how we automatically react to stressors with fight-or-flight responses, while MBSR teaches us to respond mindfully instead.",
    "Home practice is essential in MBSR. Participants are asked to practice 45 minutes per day, 6 days a week, using guided audio recordings.",
]

# ── STEP 2: Retrieve — dead-simple keyword matching ──
def retrieve(query: str, top_k: int = 3) -> list[str]:
    """Find paragraphs that contain any word from the query."""
    query_words = set(query.lower().split())
    print(f"\n🔍 RETRIEVE LOG:")
    print(f"   Query: '{query}'")
    print(f"   Query words: {query_words}")
    scored = []
    for i, para in enumerate(KNOWLEDGE_BASE):
        # Count how many query words appear in this paragraph
        para_lower = para.lower()
        matched_words = [w for w in query_words if w in para_lower]
        score = len(matched_words)
        if score > 0:
            scored.append((score, para))
            print(f"   ✅ Para {i+1}: score={score}, matched={matched_words} → {para[:60]}...")
        else:
            print(f"   ❌ Para {i+1}: no match → {para[:60]}...")
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [para for _, para in scored[:top_k]]
    print(f"   📋 Returning top {len(results)} of {len(scored)} matches")
    return results

# ── STEP 3 & 4: Augment + Generate ──
def ask(question: str) -> str:
    # Retrieve relevant context
    context_chunks = retrieve(question)

    if context_chunks:
        context_str = "\n\n".join(f"[Source {i+1}]: {c}" for i, c in enumerate(context_chunks))
    else:
        context_str = "(No relevant information found in knowledge base.)"

    # Augment: build the prompt with context
    prompt = f"""Answer the question using ONLY the provided context. 
If the context doesn't contain the answer, say "I don't have that information."

CONTEXT:
{context_str}

QUESTION: {question}

ANSWER:"""

    print (f"\n📝 PROMPT SENT TO LLM:\n{prompt}\n")
    # Generate
    response = model.generate_content(prompt)
    return response.text

# ── CLI Loop ──
print("=" * 60)
print("EXAMPLE 2: Simple Keyword RAG")
print("=" * 60)
print("Knowledge base: 8 paragraphs about MBSR")
print("Retrieval: keyword matching")
print("Try: 'What is MBSR?' or 'Tell me about body scan'\n")
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

    # Show what was retrieved (educational!)
    retrieved = retrieve(q)
    print(f"\n📎 Retrieved {len(retrieved)} chunks:")
    for i, chunk in enumerate(retrieved):
        print(f"   [{i+1}] {chunk[:80]}...")

    answer = ask(q)
    print(f"\nAI (with RAG): {answer}\n")
