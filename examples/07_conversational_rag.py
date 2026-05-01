"""
Example 7: Conversational RAG with Chat Memory
================================================
CONCEPT: Examples 2-6 treat each question independently. But users ask follow-ups:
         "What is the body scan?" → "How long does it take?" → "When do they learn it?"
         
         Conversational RAG solves this by:
           1. Rewriting the query using chat history (so "it" becomes "body scan")
           2. Retrieving with the rewritten query (better matches)
           3. Including recent chat history in the generation prompt

NEW IDEAS:
  - QUERY REWRITING:   LLM rewrites ambiguous queries using conversation context
  - CHAT MEMORY:       Last N turns stored and included in the prompt
  - CONTEXT WINDOW:    We're managing prompt size — history + retrieved chunks
                       must fit within the model's context window

THIS IS how ChatGPT + retrieval plugins, Perplexity, etc. work under the hood.

RUN:  python3 examples/07_conversational_rag.py
"""
import os, sys, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(override=True)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")

# ── Knowledge base (same MBSR chunks) ──
CHUNKS = [
    "MBSR stands for Mindfulness-Based Stress Reduction, developed by Jon Kabat-Zinn at UMass Medical Center in 1979. The program runs for 8 weeks with weekly 2.5-hour group sessions.",
    "The raisin exercise in Session 1 demonstrates autopilot behavior. Participants eat a single raisin with full attention to appearance, texture, smell, and taste. This shows how we miss daily experience.",
    "The body scan is a 45-minute practice of systematically bringing attention to each body part from toes to head. It builds interoceptive awareness and is introduced in Session 1.",
    "Sitting meditation begins with breath awareness (10-15 min in Session 2) and gradually extends to 25+ minutes by Session 4, eventually including choiceless awareness in Session 5.",
    "Gentle yoga and walking meditation are mindful movement practices introduced in Session 3. The focus is on attention and body awareness, not physical performance.",
    "The stress reaction cycle: stimulus → automatic fight-or-flight → habitual reaction. Mindfulness inserts a pause, enabling conscious response instead. Covered in Session 4.",
    "The STOP technique (Session 5): Stop what you're doing, Take a breath, Observe your experience, Proceed with awareness. A micro-practice for any stressful moment.",
    "The all-day silent retreat in Session 6 (week 6) runs 9am-4pm. All practices are combined. Many participants describe it as a turning point. Loving-kindness meditation is introduced.",
    "Session 7 focuses on maintaining practice: choosing practices, creating routines, informal mindfulness. Exercise, sleep, nutrition, and social connection support well-being.",
    "Session 8 reviews the entire program. Resources provided: books, apps, local groups. Key insight: mindfulness is not something you finish — it deepens over a lifetime.",
    "Home practice is 45 minutes per day, 6 days per week. Guided audio recordings support practice. Participants log their practice in journals and share experiences in group.",
    "Research shows MBSR reduces anxiety, depression, chronic pain symptoms, and improves immune function. Over 30,000 people have completed the program at UMass alone.",
]

# ── Build embedding index ──
def get_embedding(text: str, task: str = "retrieval_document") -> np.ndarray:
    result = genai.embed_content(model="models/gemini-embedding-2", content=text, task_type=task)
    return np.array(result['embedding'])

print("Building embedding index...")
chunk_embeddings = np.array([get_embedding(c) for c in CHUNKS])
print(f"Indexed {len(CHUNKS)} chunks\n")

# ── Retrieval ──
def retrieve(query: str, top_k: int = 3) -> list[tuple[float, str]]:
    query_emb = get_embedding(query, task="retrieval_query")
    scores = np.dot(chunk_embeddings, query_emb) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_emb)
    )
    top_idx = scores.argsort()[::-1][:top_k]
    return [(scores[i], CHUNKS[i]) for i in top_idx]

# ── NEW: Query Rewriting ──
def rewrite_query(user_query: str, chat_history: list[dict]) -> str:
    """Use the LLM to rewrite an ambiguous query using chat history."""
    if not chat_history:
        return user_query  # No history → no rewriting needed

    history_str = "\n".join(
        f"{'User' if turn['role'] == 'user' else 'AI'}: {turn['content']}"
        for turn in chat_history[-6:]  # last 3 exchanges
    )

    rewrite_prompt = f"""Given the conversation history and the latest user query, 
rewrite the query to be standalone and specific. Replace pronouns and references 
with their actual subjects. If the query is already clear, return it unchanged.

CONVERSATION HISTORY:
{history_str}

LATEST QUERY: {user_query}

REWRITTEN QUERY (just the query, nothing else):"""

    response = model.generate_content(rewrite_prompt)
    rewritten = response.text.strip().strip('"')
    return rewritten

# ── Generate with chat history ──
def ask(question: str, chat_history: list[dict]) -> str:
    # Step 1: Rewrite query for better retrieval
    rewritten = rewrite_query(question, chat_history)

    # Step 2: Retrieve with rewritten query
    results = retrieve(rewritten)
    context_str = "\n\n".join(f"[{s:.3f}] {c}" for s, c in results)

    # Step 3: Build prompt with history + context
    history_str = ""
    if chat_history:
        recent = chat_history[-6:]  # last 3 exchanges
        history_str = "RECENT CONVERSATION:\n" + "\n".join(
            f"{'User' if t['role'] == 'user' else 'AI'}: {t['content']}"
            for t in recent
        ) + "\n\n"

    prompt = f"""You are an MBSR expert answering questions about the MBSR Handbook.
Use the provided context and conversation history to give accurate, helpful answers.
If the context doesn't contain the answer, say so.

{history_str}RETRIEVED CONTEXT:
{context_str}

CURRENT QUESTION: {question}

ANSWER:"""

    response = model.generate_content(prompt)
    return response.text, rewritten, results

# ── CLI ──
print("=" * 60)
print("EXAMPLE 7: Conversational RAG with Memory")
print("=" * 60)
print("This RAG remembers your conversation! Ask follow-up questions:")
print("  1. 'What is the body scan?'")
print("  2. 'How long does it take?'     ← 'it' = body scan (rewritten!)")
print("  3. 'When is it introduced?'     ← still about body scan")
print("\nType 'history' to see chat memory, 'clear' to reset, 'quit' to exit.\n")

chat_history: list[dict] = []

while True:
    try:
        q = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!"); break
    if not q:
        continue
    if q.lower() in ("quit", "exit", "q"):
        break

    if q.lower() == "history":
        if not chat_history:
            print("\n(No history yet)\n")
        else:
            print(f"\n── Chat history ({len(chat_history)} messages) ──")
            for t in chat_history:
                print(f"  {'User' if t['role'] == 'user' else 'AI  '}: {t['content'][:100]}...")
            print()
        continue

    if q.lower() == "clear":
        chat_history.clear()
        print("Chat history cleared.\n")
        continue

    answer, rewritten, results = ask(q, chat_history)

    # Show rewriting if it changed
    if rewritten.lower() != q.lower():
        print(f"\n🔄 Rewritten query: \"{rewritten}\"")

    print(f"📎 Retrieved {len(results)} chunks (top: {results[0][0]:.3f})")
    print(f"\nAI: {answer}\n")

    # Store in history
    chat_history.append({"role": "user", "content": q})
    chat_history.append({"role": "assistant", "content": answer})
