"""
Example 4: TF-IDF Retrieval — Statistical Text Similarity
===========================================================
CONCEPT: Keyword matching (Example 2-3) treats all words equally.
         TF-IDF (Term Frequency - Inverse Document Frequency) is smarter:
           - TF:  How often a word appears in a chunk (frequent = relevant)
           - IDF: How rare a word is across ALL chunks (rare = more important)
           - TF × IDF = words that are both frequent in a chunk AND rare overall

NEW IDEA: This is the bridge between keyword search and embeddings.
          TF-IDF still uses exact word matching, but it WEIGHTS words by
          importance. "meditation" in every chunk = low IDF = less useful.
          "raisin" in only one chunk = high IDF = very discriminating.

         Uses scikit-learn's TfidfVectorizer + cosine similarity.

RUN:  python3 examples/04_tfidf_retrieval.py
"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv(override=True)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")

# ── Knowledge base (same MBSR content, pre-chunked by session) ──
CHUNKS = [
    "Session 1 introduces mindfulness through the raisin exercise. Participants eat a single raisin with full attention, noticing appearance, texture, smell, and taste. This reveals how much daily experience we miss on autopilot. The body scan meditation is introduced as homework with guided audio recordings.",
    "Session 2 explores how perceptions shape experience. The difference between direct sensory experience and mental stories is discussed. Sitting meditation focusing on breath awareness for 10-15 minutes is introduced. Thoughts are not facts — they are mental events. Stress physiology and the fight-or-flight response are covered.",
    "Session 3 introduces gentle yoga as mindful movement. The emphasis is on paying attention, not flexibility. Walking meditation is taught: walking slowly while noticing lifting, moving, placing of each foot. The pleasant events calendar is assigned as homework.",
    "Session 4 covers the stress reaction cycle. Automatic reactive patterns and how mindfulness creates a pause between stimulus and response. The concept of responding versus reacting is central. Sitting meditation extends to 20-25 minutes. The unpleasant events calendar tracks reactions to difficulties.",
    "Session 5 teaches skillful stress responding. The STOP technique: Stop, Take a breath, Observe, Proceed. This micro-practice works in any stressful moment. Difficult communication patterns are explored. Sitting meditation includes choiceless awareness — open, receptive attention.",
    "Session 6 is the all-day silent retreat from 9am to 4pm. All practices are combined: body scan, sitting meditation, yoga, walking meditation, eating meditation. Many participants call this a turning point. Loving-kindness meditation is introduced for the first time.",
    "Session 7 focuses on integration and maintaining practice after the program. Topics include choosing practices, creating sustainable routines, and informal mindfulness. Exercise, sleep, nutrition, and social connection support well-being. Mindful listening and communication are revisited.",
    "Session 8 is the final review session. Participants share learnings and changed relationships with stress. Resources are provided: books, apps, local meditation groups. The key insight: mindfulness is not something you finish — it deepens over a lifetime.",
]

# ── TF-IDF Index ──
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(CHUNKS)
feature_names = vectorizer.get_feature_names_out()

def retrieve_tfidf(query: str, top_k: int = 3) -> list[tuple[float, str]]:
    """Retrieve chunks by TF-IDF cosine similarity."""
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_k]
    return [(scores[i], CHUNKS[i]) for i in top_indices if scores[i] > 0]

def ask(question: str) -> str:
    results = retrieve_tfidf(question)
    if results:
        context_str = "\n\n".join(f"[Score: {score:.3f}] {chunk}" for score, chunk in results)
    else:
        context_str = "(No relevant chunks found.)"

    prompt = f"""Answer using ONLY the provided context. If insufficient, say so.

CONTEXT:
{context_str}

QUESTION: {question}
ANSWER:"""
    return model.generate_content(prompt).text

# ── CLI ──
print("=" * 60)
print("EXAMPLE 4: TF-IDF Retrieval")
print("=" * 60)
print(f"Index: {len(CHUNKS)} chunks, {len(feature_names)} unique terms")
print(f"Top TF-IDF terms: {', '.join(feature_names[:10])}...")
print("\nTry: 'What is the STOP technique?' or 'Tell me about the silent retreat'")
print("Type 'vocab' to see TF-IDF vocabulary, 'quit' to exit.\n")

while True:
    try:
        q = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!"); break
    if not q:
        continue
    if q.lower() in ("quit", "exit", "q"):
        break

    if q.lower() == "vocab":
        print(f"\nTF-IDF vocabulary ({len(feature_names)} terms):")
        print(", ".join(feature_names))
        print()
        continue

    results = retrieve_tfidf(q)
    print(f"\n📎 TF-IDF retrieved {len(results)} chunks:")
    for i, (score, chunk) in enumerate(results):
        print(f"   [{i+1}] score={score:.3f} → {chunk[:90]}...")

    answer = ask(q)
    print(f"\nAI: {answer}\n")
