"""
Example 9: Conversational RAG with Pinecone Vector Database
============================================================
CONCEPT: Use the Pinecone index created by setup_pinecone.py for retrieval.
         Pinecone handles all embedding (both at index time and at query time),
         so no local embedding model is needed.

HOW IT DIFFERS FROM EXAMPLE 8:
  - Example 8: loads JSONL → builds local numpy embeddings → cosine search
  - Example 9: connects to Pinecone index → sends query text → Pinecone
               embeds the query and searches server-side → returns top chunks

FEATURES:
  - No local embedding computation (Pinecone does it all)
  - Adaptive query complexity detection (top_k 2-5)
  - Query rewriting using chat history for better retrieval
  - Full conversational memory (no truncation)
  - Enhanced system prompt preserving MBSR philosophical depth

PREREQUISITES:
  Run setup_pinecone.py first to populate the index:
    python3 setup_pinecone.py --create

RUN:
  python3 examples/09_conversational_rag_with_pinecone_db.py
"""
import os
import sys

# Allow imports from the project root (for .env loading)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai

load_dotenv(override=True)

# ── Configuration ──────────────────────────────────────────────────────────────
# These must match the values used in setup_pinecone.py when creating the index.
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
INDEX_NAME         = "mbsr-rag-index"
NAMESPACE          = "mbsr"   # namespace used during upsert_records in setup script

GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")

# Initialise clients
genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-2.5-flash")


# ── Pinecone initialisation ────────────────────────────────────────────────────
def init_pinecone_index():
    """
    Connect to the existing Pinecone index.

    The index must already exist (created by setup_pinecone.py --create).
    Because the index was created with create_index_for_model, it knows
    which embedding model to use — we never call an embedding API ourselves.
    """
    if not PINECONE_API_KEY:
        print("❌ PINECONE_API_KEY not found in .env")
        sys.exit(1)

    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Verify index exists before continuing
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"❌ Index '{INDEX_NAME}' not found.")
        print("   Run:  python3 setup_pinecone.py --create")
        sys.exit(1)

    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    print(f"✓ Connected to Pinecone index '{INDEX_NAME}'")
    print(f"  Vectors stored: {stats.total_vector_count}\n")
    return index


# ── Query Complexity Detection ─────────────────────────────────────────────────
def estimate_query_complexity(query: str) -> int:
    """
    Estimate query complexity and return an adaptive top_k value.

    Simple queries need fewer chunks; complex multi-part questions need more
    context to give a complete answer.

    Returns:
        2  for simple queries  (≤3 words, no multi-question)
        4  for medium queries  (4-8 words or 1 question mark)
        5  for complex queries (9+ words or multiple questions)
    """
    word_count     = len(query.split())
    question_count = query.count("?")
    complexity     = word_count + (question_count * 2)

    if complexity < 4:
        return 2
    elif complexity < 10:
        return 4
    else:
        return 5


# ── Retrieval via Pinecone ─────────────────────────────────────────────────────
def retrieve(query: str, index, top_k: int = None) -> list[dict]:
    """
    Search Pinecone for the most relevant MBSR chunks.

    HOW IT WORKS:
    - We send the raw query text to index.search().
    - Pinecone embeds it using the same model that was used at upload time
      (llama-text-embed-v2, set in create_index_for_model).
    - Pinecone returns the top_k closest vectors along with their stored fields.

    Args:
        query:  The (possibly rewritten) user question.
        index:  Active Pinecone Index object.
        top_k:  Number of results to fetch. Determined adaptively if None.

    Returns:
        List of result dicts, each containing:
          - score:      cosine similarity score (0-1)
          - chunk_text: the original MBSR passage
          - source:     source file name
          - page:       page number
    """
    if top_k is None:
        top_k = estimate_query_complexity(query)

    # index.search() accepts a plain text query for integrated-embedding indexes.
    # Pinecone embeds it server-side using the same model as the stored vectors.
    results = index.search(
        namespace=NAMESPACE,
        query={"inputs": {"text": query}, "top_k": top_k}
    )

    # Normalise results into a consistent dict format for the rest of the code
    hits = []
    for match in results.result.hits:
        fields = match.fields  # metadata stored alongside the vector
        hits.append({
            "score":      match._score,
            "chunk_text": fields.get("chunk_text", ""),
            "source":     fields.get("source", "unknown"),
            "page":       fields.get("page", "?"),
        })
    return hits


# ── Query Rewriting ────────────────────────────────────────────────────────────
def rewrite_query(user_query: str, chat_history: list[dict]) -> str:
    """
    Use the LLM to rewrite an ambiguous query into a standalone question.

    WHY: Users often ask follow-up questions like "What about the second one?"
         or "Can you elaborate?". These pronouns/references can't be resolved
         without context. The rewritten query improves Pinecone retrieval.

    Uses full chat history (no character truncation) to preserve context
    across long conversations.
    """
    if not chat_history:
        return user_query

    # Use last 6 messages (3 exchanges) for context
    history_str = "\n".join(
        f"{'User' if t['role'] == 'user' else 'AI'}: {t['content']}"
        for t in chat_history[-6:]
    )

    prompt = f"""Given the conversation history and the latest user query,
rewrite the query to be standalone and specific. Replace pronouns and references
with their actual subjects. If the query is already clear, return it unchanged.

CONVERSATION HISTORY:
{history_str}

LATEST QUERY: {user_query}

REWRITTEN QUERY (just the query, nothing else):"""

    response = llm.generate_content(prompt)
    return response.text.strip().strip('"')


# ── Answer Generation ──────────────────────────────────────────────────────────
def ask(question: str, chat_history: list[dict], index) -> tuple[str, str, list]:
    """
    Full RAG pipeline: rewrite → retrieve → generate.

    Steps:
    1. Rewrite the query using chat history (resolves pronouns/references).
    2. Retrieve top chunks from Pinecone (server-side embedding).
    3. Build a prompt combining history + retrieved context + question.
    4. Generate an answer with the LLM, preserving MBSR philosophical depth.

    Returns:
        answer    - LLM-generated response string
        rewritten - the rewritten query (for display)
        hits      - list of retrieved chunk dicts (for display)
    """
    # Step 1: Rewrite query for better retrieval accuracy
    rewritten = rewrite_query(question, chat_history)

    # Step 2: Retrieve relevant MBSR chunks from Pinecone
    hits = retrieve(rewritten, index)

    # Build context block from retrieved chunks
    context_str = "\n\n".join(
        f"[Score: {h['score']:.3f} | {h['source']} p.{h['page']}]\n{h['chunk_text']}"
        for h in hits
    )

    # Step 3: Build prompt with recent history + context
    history_str = ""
    if chat_history:
        recent = chat_history[-6:]  # Last 3 exchanges
        history_str = "RECENT CONVERSATION:\n" + "\n".join(
            f"{'User' if t['role'] == 'user' else 'AI'}: {t['content']}"
            for t in recent
        ) + "\n\n"

    prompt = f"""You are an MBSR (Mindfulness-Based Stress Reduction) expert
answering questions about the MBSR Handbook and mindfulness practices.

IMPORTANT GUIDELINES:
1. Ground all answers in the provided context - cite specific passages when possible.
2. Capture both STRUCTURE and ESSENCE:
   - Structure: Steps, sequences, mechanics (e.g., "start with left foot, move upward")
   - Essence: Experiential insights, philosophical depth, awareness goals
     (e.g., "dissolving subject-object separation", "non-judgmental awareness")
3. Be COMPLETE: If context mentions multiple aspects (body parts, emotional/philosophical
   insights), include all of them — do not reduce to just the mechanical steps.
4. Preserve the TONE and SPIRIT of MBSR: Mindfulness is not just technique — it is
   a way of being. Honour that depth in every answer.
5. If the context does not contain relevant information, say so clearly and
   avoid speculation.

{history_str}RETRIEVED CONTEXT:
{context_str}

CURRENT QUESTION: {question}

ANSWER:"""

    response = llm.generate_content(prompt)
    return response.text, rewritten, hits


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    """Interactive conversational RAG loop backed by Pinecone."""
    print("=" * 70)
    print("EXAMPLE 9: Conversational RAG with Pinecone Vector Database")
    print("=" * 70)

    # Verify credentials
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not found in .env")
        sys.exit(1)

    # Connect to Pinecone
    index = init_pinecone_index()

    print("Commands:")
    print("  - Type your question to get an MBSR-based answer")
    print("  - Type 'history' to see chat memory")
    print("  - Type 'clear'   to reset chat history")
    print("  - Type 'quit'    to exit\n")

    chat_history: list[dict] = []

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not q:
            continue

        if q.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if q.lower() == "history":
            if not chat_history:
                print("\n(No chat history yet)\n")
            else:
                print(f"\n── Chat History ({len(chat_history)} messages) ──")
                for i, turn in enumerate(chat_history, 1):
                    role    = "User" if turn["role"] == "user" else "AI  "
                    content = turn["content"][:100] + "..." if len(turn["content"]) > 100 else turn["content"]
                    print(f"  {i}. {role}: {content}")
                print()
            continue

        if q.lower() == "clear":
            chat_history.clear()
            print("Chat history cleared.\n")
            continue

        # ── Get answer ──
        answer, rewritten, hits = ask(q, chat_history, index)

        # Show query rewriting if it changed
        if rewritten.lower() != q.lower():
            print(f"\n🔄 Rewritten query: \"{rewritten}\"")

        # Show retrieval info
        print(f"📎 Retrieved {len(hits)} chunks from Pinecone")
        if hits:
            top = hits[0]
            print(f"   Top match score: {top['score']:.3f}")
            print(f"   Source: {top['source']} (page {top['page']})")

        print(f"\nAI: {answer}\n")

        # Append to history
        chat_history.append({"role": "user",      "content": q})
        chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
