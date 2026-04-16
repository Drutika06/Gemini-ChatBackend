"""
Example 5b: Sentence ↔ Question Similarity — How RAG Retrieval Works
======================================================================
CONCEPT: In RAG, we match a USER QUESTION to stored SENTENCES/CHUNKS.
         This demo shows exactly what happens under the hood:
           1. We embed a set of short knowledge sentences
           2. You ask a question → it gets embedded too
           3. Cosine similarity ranks which sentences are closest
           4. The top matches become the "context" for the LLM

         You'll see that embeddings match MEANING, not keywords:
           Q: "How do I calm down quickly?"
           → matches "The STOP technique: Stop, Take a breath, Observe, Proceed."
           Even though they share ZERO words!

         This is the core magic of embedding-based RAG.

RUN:  python3 examples/05b_sentence_similarity_demo.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(override=True)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ── Embedding helpers ──
def get_embedding(text: str, task: str = "retrieval_document") -> np.ndarray:
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type=task
    )
    return np.array(result['embedding'])

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

BASELINE = 0.80

def rescale(score: float) -> float:
    return max(0.0, min(1.0, (score - BASELINE) / (1.0 - BASELINE)))

def similarity_bar(score: float, width: int = 25) -> str:
    scaled = rescale(score)
    filled = int(scaled * width)
    return "█" * filled + "░" * (width - filled)

def interpret(score: float) -> str:
    if score >= 0.97:
        return "exact match"
    elif score >= 0.93:
        return "strong match"
    elif score >= 0.87:
        return "related"
    elif score >= BASELINE:
        return "unrelated"
    else:
        return "noise"

# ── Knowledge sentences (small, clear facts) ──
SENTENCES = [
    "The Earth revolves around the Sun and takes 365 days to complete one orbit.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Python is a popular programming language used for web development and data science.",
    "The STOP technique helps manage stress: Stop, Take a breath, Observe, Proceed.",
    "Photosynthesis is the process by which plants convert sunlight into food.",
    "The Great Wall of China is over 13,000 miles long and was built over many centuries.",
    "Regular exercise improves cardiovascular health and reduces anxiety.",
    "టమాటా ఒక పండు, కానీ వంటలో కూరగాయగా వాడతారు.",  # Tomato is a fruit but used as vegetable
    "నీరు 100 డిగ్రీల సెల్సియస్ వద్ద మరుగుతుంది.",  # Water boils at 100°C
    "భూమి సూర్యుని చుట్టూ 365 రోజుల్లో ఒక చుట్టు తిరుగుతుంది.",  # Earth orbits Sun in 365 days
]

# ── Pre-built questions to demo the concept ──
DEMO_QUESTIONS = [
    "How long does Earth take to go around the Sun?",
    "At what temperature does water start boiling?",
    "How do I calm down when I'm stressed?",
    "What language is good for data science?",
    "How do plants make their own food?",
    "భూమి సూర్యుని చుట్టూ ఎన్ని రోజుల్లో తిరుగుతుంది?",  # How many days for Earth to orbit Sun?
    "నీరు ఎంత ఉష్ణోగ్రత వద్ద మరుగుతుంది?",  # At what temp does water boil?
]

# ── Build index ──
print("Embedding sentences...", end=" ", flush=True)
sentence_embeddings = [get_embedding(s, "retrieval_document") for s in SENTENCES]
print(f"done. ({len(SENTENCES)} sentences, dim={len(sentence_embeddings[0])})\n")

def rank_sentences(question: str) -> list[tuple[float, int, str]]:
    """Rank all sentences by similarity to the question."""
    q_emb = get_embedding(question, "retrieval_query")
    results = []
    for i, (s_emb, s_text) in enumerate(zip(sentence_embeddings, SENTENCES)):
        sim = cosine_sim(q_emb, s_emb)
        results.append((sim, i, s_text))
    results.sort(key=lambda x: x[0], reverse=True)
    return results

# ── CLI ──
print("=" * 65)
print("SENTENCE ↔ QUESTION SIMILARITY DEMO")
print("=" * 65)
print("See how RAG retrieval matches questions to knowledge!\n")
print("Knowledge base:")
for i, s in enumerate(SENTENCES):
    print(f"  [{i+1:2d}] {s[:80]}{'...' if len(s) > 80 else ''}")
print()
print("Commands:")
print("  demo    — Run pre-built questions and see ranked matches")
print("  ask     — Type your own question")
print("  vs      — Compare two sentences directly")
print("  quit    — Exit\n")

while True:
    try:
        cmd = input(">> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!"); break
    if not cmd:
        continue
    if cmd in ("quit", "exit", "q"):
        break

    # ── Demo: run all pre-built questions ──
    if cmd == "demo":
        for question in DEMO_QUESTIONS:
            print(f"\n{'─' * 65}")
            print(f"  ❓ {question}")
            print(f"{'─' * 65}")
            ranked = rank_sentences(question)
            for rank, (sim, idx, text) in enumerate(ranked):
                bar = similarity_bar(sim)
                tag = interpret(sim)
                marker = " ✅" if rank == 0 else ""
                preview = text[:60] + "..." if len(text) > 60 else text
                print(f"  {sim:.3f} {bar} [{tag:>10}] [{idx+1:2d}] {preview}{marker}")
            print()
        print("Notice how questions match the RIGHT sentence even without shared keywords!")
        print("Also notice Telugu questions matching both Telugu AND English answers.\n")
        continue

    # ── Ask your own question ──
    if cmd == "ask":
        question = input("  Your question: ").strip()
        if not question:
            continue
        print(f"\n  Embedding question...", end=" ", flush=True)
        ranked = rank_sentences(question)
        print("done.\n")

        print(f"  ❓ {question}\n")
        print(f"  {'Score':>6}  {'Bar':<25}  {'Match':>10}  Sentence")
        print(f"  {'─'*6}  {'─'*25}  {'─'*10}  {'─'*40}")
        for rank, (sim, idx, text) in enumerate(ranked):
            bar = similarity_bar(sim)
            tag = interpret(sim)
            marker = " ← BEST" if rank == 0 else ""
            preview = text[:55] + "..." if len(text) > 55 else text
            print(f"  {sim:.3f}  {bar}  [{tag:>10}]  [{idx+1:2d}] {preview}{marker}")
        
        print(f"\n  In RAG, the top 2-3 results would be stuffed into the LLM prompt as context.\n")
        continue

    # ── Compare two sentences directly ──
    if cmd == "vs":
        print("  Pick two sentence numbers to compare:")
        for i, s in enumerate(SENTENCES):
            print(f"    [{i+1:2d}] {s[:70]}{'...' if len(s) > 70 else ''}")
        try:
            a = int(input("  Sentence A (number): ").strip()) - 1
            b = int(input("  Sentence B (number): ").strip()) - 1
            if not (0 <= a < len(SENTENCES) and 0 <= b < len(SENTENCES)):
                print("  Invalid numbers.\n")
                continue
        except ValueError:
            print("  Enter numbers.\n")
            continue

        sim = cosine_sim(sentence_embeddings[a], sentence_embeddings[b])
        bar = similarity_bar(sim)
        tag = interpret(sim)
        print(f"\n  A: {SENTENCES[a][:70]}...")
        print(f"  B: {SENTENCES[b][:70]}...")
        print(f"\n  Cosine similarity: {sim:.4f} (raw)")
        print(f"  Rescaled: {rescale(sim):.2%}")
        print(f"  {bar}  [{tag}]\n")
        continue

    print("  Unknown command. Try: demo, ask, vs, quit\n")
