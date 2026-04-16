"""
Example 5: Embedding-Based RAG — Semantic Search
==================================================
CONCEPT: TF-IDF (Example 4) still relies on exact word overlap.
         EMBEDDINGS capture MEANING: "stress reduction" and "calming technique"
         are close in embedding space even though they share zero words.

         How it works:
           1. Each chunk → embedding vector (768+ dimensions) via Gemini API
           2. User query → embedding vector
           3. Cosine similarity finds chunks closest in meaning
           4. Top chunks → stuffed into the LLM prompt

NEW IDEA: Embeddings are the standard retrieval method in modern RAG.
          They understand synonyms, paraphrasing, and conceptual similarity.

RUN:  python3 examples/05_embedding_rag.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(override=True)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")

# ── Knowledge base ──
CHUNKS = [
    "MBSR stands for Mindfulness-Based Stress Reduction, developed by Jon Kabat-Zinn at UMass Medical Center in 1979. The program runs for 8 weeks with weekly 2.5-hour group sessions.",
    "The raisin exercise in Session 1 demonstrates autopilot behavior. Participants eat a single raisin with full attention to appearance, texture, smell, and taste.",
    "The body scan is a 45-minute practice of systematically bringing attention to each body part from toes to head. It builds interoceptive awareness.",
    "Sitting meditation begins with breath awareness (10-15 min) and gradually extends to 25+ minutes, eventually including choiceless awareness.",
    "Gentle yoga and walking meditation are mindful movement practices. The focus is on attention and body awareness, not physical performance.",
    "The stress reaction cycle: stimulus → automatic fight-or-flight → habitual reaction. Mindfulness inserts a pause, enabling conscious response instead.",
    "The STOP technique is a micro-practice: Stop what you're doing, Take a breath, Observe your experience, Proceed with awareness.",
    "The all-day silent retreat in week 6 combines all practices. Participants often describe it as a transformative turning point in their practice.",
    "Loving-kindness meditation (metta) involves silently repeating phrases of goodwill toward self and others: May I be happy, may I be healthy, may I be safe.",
    "Home practice is 45 minutes per day, 6 days per week. Guided audio recordings support the practice. Consistency matters more than perfection.",
]

# ── Build embedding index ──
print("Building embedding index...")

def get_embedding(text: str) -> np.ndarray:
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )
    return np.array(result['embedding'])

def get_query_embedding(text: str) -> np.ndarray:
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_query"
    )
    return np.array(result['embedding'])

# Pre-compute embeddings for all chunks
chunk_embeddings = np.array([get_embedding(c) for c in CHUNKS])
print(f"Indexed {len(CHUNKS)} chunks, embedding dim = {chunk_embeddings.shape[1]}")

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query: str, top_k: int = 3) -> list[tuple[float, str]]:
    query_emb = get_query_embedding(query)
    scores = [cosine_sim(query_emb, ce) for ce in chunk_embeddings]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(scores[i], CHUNKS[i]) for i in top_indices]

def ask(question: str) -> str:
    results = retrieve(question)
    context_str = "\n\n".join(f"[Similarity: {s:.3f}] {c}" for s, c in results)
    prompt = f"""Answer using ONLY the provided context. If insufficient, say so.

CONTEXT:
{context_str}

QUESTION: {question}
ANSWER:"""
    return model.generate_content(prompt).text

# ── CLI ──
print("\n" + "=" * 60)
print("EXAMPLE 5: Embedding-Based RAG (Semantic Search)")
print("=" * 60)
print("Now using Gemini embeddings for retrieval — understands MEANING!")
print("\nTry these to see semantic search vs keyword:")
print("  • 'How do I calm down quickly?'  (matches STOP technique)")
print("  • 'body awareness practice'      (matches body scan)")
print("  • 'being kind to yourself'        (matches loving-kindness)")
print("\nType 'compare <query>' to see TF-IDF vs embedding results side by side.")
print("Type 'quit' to exit.\n")

# Optional: TF-IDF comparison
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
tfidf_vec = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vec.fit_transform(CHUNKS)

while True:
    try:
        q = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!"); break
    if not q:
        continue
    if q.lower() in ("quit", "exit", "q"):
        break

    if q.lower().startswith("compare "):
        query = q[8:]
        # TF-IDF results
        tfidf_scores = sklearn_cosine(tfidf_vec.transform([query]), tfidf_matrix).flatten()
        tfidf_top = tfidf_scores.argsort()[::-1][:3]
        # Embedding results
        emb_results = retrieve(query, top_k=3)

        print(f"\n── Comparison for: '{query}' ──")
        print("\n🔤 TF-IDF (keyword-based):")
        for i in tfidf_top:
            print(f"   [{tfidf_scores[i]:.3f}] {CHUNKS[i][:80]}...")
        print("\n🧠 Embedding (semantic):")
        for score, chunk in emb_results:
            print(f"   [{score:.3f}] {chunk[:80]}...")
        print()
        continue

    results = retrieve(q)
    print(f"\n🧠 Semantic search — top {len(results)} chunks:")
    for i, (score, chunk) in enumerate(results):
        print(f"   [{i+1}] sim={score:.3f} → {chunk[:90]}...")

    answer = ask(q)
    print(f"\nAI: {answer}\n")
