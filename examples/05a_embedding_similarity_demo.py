"""
Example 5a: Embedding Similarity Demo — See Meaning as Numbers
================================================================
CONCEPT: Embeddings turn words into vectors. Similar meanings → similar vectors.
         This demo lets you compare ANY words/phrases and see their cosine similarity.

         You'll see that:
           - "big" and "large" score ~0.97  (true synonyms)
           - "dog" and "puppy" score ~0.93  (strongly related)
           - "dog" and "cat" score ~0.88   (same category)
           - "sun" and "banana" score ~0.83 (unrelated — baseline noise)

         IMPORTANT: Dense embeddings have a HIGH BASELINE (~0.80).
         The raw cosine number is NOT a percentage of similarity!
         The entire meaningful range is compressed into roughly [0.80, 1.0]:

           0.97+  = near-synonym (big ↔ large)
           0.93+  = strongly related (dog ↔ puppy)
           0.87+  = same category (dog ↔ cat)
           <0.87  = baseline / unrelated (sun ↔ banana)

         We rescale to [0, 1] for display so the bars actually make sense.

RUN:  python3 examples/05a_embedding_similarity_demo.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(override=True)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ── Embedding helper ──
def get_embedding(text: str) -> np.ndarray:
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type="semantic_similarity"
    )
    return np.array(result['embedding'])

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Dense embeddings have a high baseline (~0.80). We rescale the
# meaningful range [BASELINE, 1.0] → [0, 1] so the bar is useful.
BASELINE = 0.80

def rescale(score: float) -> float:
    """Rescale from [BASELINE, 1.0] → [0.0, 1.0] for display."""
    return max(0.0, min(1.0, (score - BASELINE) / (1.0 - BASELINE)))

def similarity_bar(score: float, width: int = 30) -> str:
    scaled = rescale(score)
    filled = int(scaled * width)
    return "█" * filled + "░" * (width - filled)

def interpret(score: float) -> str:
    """Interpret cosine similarity using calibrated thresholds for dense embeddings."""
    if score >= 0.97:
        return "Near-synonym — almost identical meaning"
    elif score >= 0.93:
        return "Strongly related — very close in meaning"
    elif score >= 0.87:
        return "Same category — clearly related concepts"
    elif score >= BASELINE:
        return "Baseline — unrelated (this is just embedding noise)"
    else:
        return "Below baseline — no meaningful relationship"

# ── Pre-built word groups to demonstrate the concept ──
DEMO_GROUPS = [
    {
        "label": "Celestial bodies",
        "words": ["sun", "moon", "star", "planet", "car", "banana"],
    },
    {
        "label": "Emotions",
        "words": ["happy", "joyful", "sad", "angry", "excited", "table"],
    },
    {
        "label": "Animals",
        "words": ["dog", "puppy", "cat", "wolf", "laptop", "fish"],
    },
    {
        "label": "Synonyms & opposites",
        "words": ["big", "large", "huge", "small", "tiny", "purple"],
    },
    {
        "label": "Telugu words (same meaning comparison)",
        "words": ["సూర్యుడు", "చంద్రుడు", "నక్షత్రం", "కుక్క", "పిల్లి", "సంతోషం"],
    },
    {
        "label": "Telugu ↔ English (cross-language meaning)",
        "words": ["sun", "సూర్యుడు", "moon", "చంద్రుడు", "dog", "కుక్క"],
    },
    {
        "label": "Telugu emotions ↔ English emotions",
        "words": ["happy", "సంతోషం", "sad", "బాధ", "anger", "కోపం"],
    },
]

# ── CLI ──
print("=" * 60)
print("EMBEDDING SIMILARITY DEMO")
print("=" * 60)
print("See how embeddings capture meaning as numbers!")
print("⚠️  Raw cosine scores have a HIGH BASELINE (~0.80) — they are NOT percentages!")
print(f"   Bars are rescaled from [{BASELINE}, 1.0] → [0%, 100%] for clarity.\n")
print("Commands:")
print("  demo          — Run pre-built word group comparisons")
print("  pair          — Compare two words/phrases")
print("  matrix        — Compare a list of words (similarity matrix)")
print("  quit          — Exit\n")

# Cache embeddings to avoid repeated API calls
embedding_cache: dict[str, np.ndarray] = {}

def cached_embedding(text: str) -> np.ndarray:
    if text not in embedding_cache:
        embedding_cache[text] = get_embedding(text)
    return embedding_cache[text]

while True:
    try:
        cmd = input(">> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!"); break
    if not cmd:
        continue
    if cmd in ("quit", "exit", "q"):
        break

    # ── Demo mode: show pre-built comparisons ──
    if cmd == "demo":
        for group in DEMO_GROUPS:
            print(f"\n{'─' * 50}")
            print(f"  {group['label']}: {', '.join(group['words'])}")
            print(f"{'─' * 50}")

            words = group['words']
            print(f"  Embedding words...", end=" ", flush=True)
            embs = {w: cached_embedding(w) for w in words}
            print("done.\n")

            # Compare each pair with the first word
            base = words[0]
            for other in words[1:]:
                sim = cosine_sim(embs[base], embs[other])
                bar = similarity_bar(sim)
                label = interpret(sim)
                print(f"  {base:>10} ↔ {other:<10}  {sim:.3f}  {bar}  {label}")
            print()
        continue

    # ── Pair mode: compare two words ──
    if cmd == "pair":
        word1 = input("  Word/phrase 1: ").strip()
        word2 = input("  Word/phrase 2: ").strip()
        if not word1 or not word2:
            print("  Need two words.\n")
            continue

        print(f"  Embedding...", end=" ", flush=True)
        emb1 = cached_embedding(word1)
        emb2 = cached_embedding(word2)
        sim = cosine_sim(emb1, emb2)
        print("done.\n")

        bar = similarity_bar(sim)
        label = interpret(sim)
        print(f"  '{word1}' ↔ '{word2}'")
        print(f"  Cosine similarity: {sim:.4f}  (raw)")
        print(f"  Rescaled to [{BASELINE},1]: {rescale(sim):.2%}")
        print(f"  {bar}")
        print(f"  Embedding dimension: {len(emb1)}")
        print(f"  → {label}\n")
        continue

    # ── Matrix mode: NxN similarity grid ──
    if cmd == "matrix":
        raw = input("  Enter words (comma-separated): ").strip()
        words = [w.strip() for w in raw.split(",") if w.strip()]
        if len(words) < 2:
            print("  Need at least 2 words.\n")
            continue

        print(f"  Embedding {len(words)} words...", end=" ", flush=True)
        embs = [cached_embedding(w) for w in words]
        print("done.\n")

        # Print header
        max_len = max(len(w) for w in words)
        header = " " * (max_len + 2) + "  ".join(f"{w:>{max_len}}" for w in words)
        print(f"  {header}")

        # Print similarity matrix
        for i, w1 in enumerate(words):
            row = f"  {w1:>{max_len}}"
            for j, w2 in enumerate(words):
                sim = cosine_sim(embs[i], embs[j])
                row += f"  {sim:>{max_len}.3f}"
            print(row)
        print()
        continue

    print("  Unknown command. Try: demo, pair, matrix, quit\n")
