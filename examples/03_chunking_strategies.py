"""
Example 3: Chunking Strategies — Why Size Matters
===================================================
CONCEPT: Real documents are too big to fit in a prompt. We must CHUNK them.
         This example loads a text file and demonstrates 3 chunking strategies:
           1. Fixed-size chunks (naive)
           2. Fixed-size with overlap (better — preserves context at boundaries)
           3. Sentence-based chunks (respects natural boundaries)

NEW IDEA: Chunk size and overlap directly affect retrieval quality.
          Too small → fragments lose meaning. Too big → noise drowns signal.

         We still use keyword search here — embeddings come in Example 5.

RUN:  python3 examples/03_chunking_strategies.py
"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(override=True)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")

# ── Sample document (inline for simplicity) ──
DOCUMENT = """
Mindfulness-Based Stress Reduction (MBSR) Program Overview

Session 1: Introduction to Mindfulness
The first session introduces participants to the concept of mindfulness. The raisin exercise 
is used to demonstrate how we often operate on autopilot. Participants eat a single raisin 
with full attention, noticing its appearance, texture, smell, and taste. This simple exercise 
reveals how much of our daily experience we miss. The body scan meditation is also introduced 
as a homework practice. Participants receive guided audio recordings for home practice.

Session 2: Perception and Reality  
This session explores how our perceptions shape our experience. We discuss the difference 
between direct sensory experience and the stories our mind creates. The sitting meditation 
is introduced, initially focusing on breath awareness for 10-15 minutes. Participants learn 
that thoughts are not facts — they are mental events that come and go. Stress physiology is 
also introduced: the fight-or-flight response and how chronic stress affects the body.

Session 3: Mindfulness in Movement
Gentle yoga is introduced as a mindful movement practice. The emphasis is not on flexibility 
or strength, but on paying close attention to the body during movement. Participants learn 
to notice their limits without judgment. Walking meditation is also taught: walking very 
slowly while paying attention to each component of a step — lifting, moving, placing. 
The pleasant events calendar is assigned as homework.

Session 4: The Stress Reaction
This session dives deep into the stress reaction cycle. Participants learn about automatic 
reactive patterns and how mindfulness creates a pause between stimulus and response. The 
concept of "responding vs reacting" is central. Sitting meditation is extended to 20-25 
minutes. The unpleasant events calendar helps participants notice habitual reactions to 
difficult experiences.

Session 5: Stress Responding
Building on Session 4, participants explore skillful ways to respond to stress. The STOP 
technique is taught: Stop, Take a breath, Observe, Proceed. This micro-practice can be 
used in any moment of stress. Difficult communication patterns are explored. Sitting 
meditation now includes choiceless awareness — resting in open, receptive attention.

Session 6: All-Day Retreat
The day-long retreat typically runs from 9am to 4pm and is held in silence. All practices 
learned so far are woven together: body scan, sitting meditation, yoga, walking meditation, 
and eating meditation. Many participants report this as a turning point — experiencing 
extended silence and the depth of practice. Loving-kindness meditation is introduced.

Session 7: Integration  
Participants explore how to maintain their practice after the program ends. Topics include 
choosing which practices to continue, creating a sustainable routine, and using informal 
mindfulness in daily life. Exercise, sleep, nutrition, and social connection are discussed 
as foundations for well-being. Communication styles are revisited with mindful listening.

Session 8: Keeping the Momentum
The final session reviews the entire program. Participants share what they've learned and 
how their relationship to stress has changed. Resources for ongoing practice are provided, 
including books, apps, and local meditation groups. The key message: mindfulness is not 
something you "finish" — it is a way of being that deepens over a lifetime of practice.
"""

# ================================================================
# THREE CHUNKING STRATEGIES
# ================================================================

def chunk_fixed(text: str, chunk_size: int = 300) -> list[str]:
    """Strategy 1: Fixed-size character chunks. Simple but can cut mid-sentence."""
    words = text.split()
    chunks, current = [], []
    length = 0
    for word in words:
        current.append(word)
        length += len(word) + 1
        if length >= chunk_size:
            chunks.append(" ".join(current))
            current, length = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks

def chunk_overlap(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Strategy 2: Fixed-size with overlap. Overlapping windows preserve boundary context."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start
        length = 0
        while end < len(words) and length < chunk_size:
            length += len(words[end]) + 1
            end += 1
        chunks.append(" ".join(words[start:end]))
        # Move forward, but keep 'overlap' chars worth of words
        overlap_words = max(1, overlap // 5)  # rough estimate
        start = max(start + 1, end - overlap_words)
    return chunks

def chunk_sentences(text: str, max_sentences: int = 4) -> list[str]:
    """Strategy 3: Sentence-based chunks. Respects natural text boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

# ── Retrieval (same keyword approach as Example 2) ──
def retrieve(query: str, chunks: list[str], top_k: int = 3) -> list[str]:
    query_words = set(query.lower().split())
    scored = []
    for chunk in chunks:
        score = sum(1 for w in query_words if w in chunk.lower())
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]

def ask(question: str, chunks: list[str]) -> str:
    context_chunks = retrieve(question, chunks)
    context_str = "\n\n".join(f"[Chunk {i+1}]: {c}" for i, c in enumerate(context_chunks)) if context_chunks else "(No relevant chunks found.)"

    prompt = f"""Answer using ONLY the provided context. If insufficient, say so.

CONTEXT:
{context_str}

QUESTION: {question}
ANSWER:"""
    print("⏳ Calling Gemini API...")
    response = model.generate_content(prompt)
    print("✅ Gemini API responded.")
    return response.text

# ── CLI ──
print("=" * 60)
print("EXAMPLE 3: Chunking Strategies")
print("=" * 60)

strategies = {
    "1": ("Fixed-size (300 chars)", chunk_fixed(DOCUMENT, 300)),
    "2": ("Overlap (300 chars, 50 overlap)", chunk_overlap(DOCUMENT, 300, 50)),
    "3": ("Sentence-based (4 sentences)", chunk_sentences(DOCUMENT, 4)),
}

for key, (name, chunks) in strategies.items():
    print(f"  [{key}] {name} → {len(chunks)} chunks")

print("\nPick a strategy, then ask questions to see how chunking affects answers.")
print("Type 'switch' to change strategy, 'chunks' to see all chunks, 'quit' to exit.\n")

current = "3"  # default to sentence-based
print(f"Using: {strategies[current][0]}\n")

while True:
    try:
        q = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!"); break
    if not q:
        continue
    if q.lower() in ("quit", "exit", "q"):
        break

    if q.lower() == "switch":
        choice = input("Pick strategy [1/2/3]: ").strip()
        if choice in strategies:
            current = choice
            print(f"Switched to: {strategies[current][0]}\n")
        continue

    if q.lower() == "chunks":
        name, chunks = strategies[current]
        print(f"\n── {name}: {len(chunks)} chunks ──")
        for i, c in enumerate(chunks):
            print(f"\n[{i+1}] ({len(c)} chars): {c[:120]}...")
        print()
        continue

    name, chunks = strategies[current]
    retrieved = retrieve(q, chunks)
    print(f"\n📎 Retrieved {len(retrieved)} chunks ({name}):")
    for i, c in enumerate(retrieved):
        print(f"   [{i+1}] {c[:100]}...")

    answer = ask(q, chunks)
    print(f"\nAI: {answer}\n")
