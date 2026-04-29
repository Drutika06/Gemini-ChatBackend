"""
Example 8: Conversational RAG with Real MBSR Data from JSONL
============================================================
CONCEPT: Load real MBSR chunks from mbsr_chunks.jsonl, build embeddings,
         and provide conversational RAG responses with chat memory.

FEATURES:
  - Loads chunks from the real JSONL file
  - Builds semantic embeddings for all chunks
  - Query rewriting for better context understanding
  - Chat memory for conversational follow-ups
  - Semantic similarity-based retrieval

RUN:  python3 examples/08_conversational_rag_with_real_data.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(override=True)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")


# ── Load MBSR chunks from JSONL ──
def load_chunks_from_jsonl(filepath: str) -> list[dict]:
    """Load MBSR chunks from JSONL file."""
    chunks = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    chunks.append({
                        "id": chunk.get("id", ""),
                        "text": chunk.get("text", ""),
                        "metadata": chunk.get("metadata", {})
                    })
        print(f"✓ Loaded {len(chunks)} chunks from {filepath}\n")
        return chunks
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
        sys.exit(1)


# ── Build embedding index ──
def get_embedding(text: str, task: str = "retrieval_document") -> np.ndarray:
    """Generate embedding for text using Google's embedding model."""
    result = genai.embed_content(
        model="models/gemini-embedding-2",
        content=text,
        task_type=task
    )
    return np.array(result['embedding'])


def build_embedding_index(chunks: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Build embeddings for all chunks."""
    print("Building embedding index...")
    chunk_texts = [c["text"] for c in chunks]
    embeddings = np.array([get_embedding(text) for text in chunk_texts])
    print(f"✓ Built embeddings for {len(chunks)} chunks\n")
    return embeddings, chunk_texts


# ── Retrieval ──
def retrieve(
    query: str,
    embeddings: np.ndarray,
    chunk_texts: list[str],
    chunks: list[dict],
    top_k: int = 3
) -> list[tuple[float, str, dict]]:
    """Retrieve top-k chunks using cosine similarity."""
    query_emb = get_embedding(query, task="retrieval_query")
    
    # Cosine similarity
    scores = np.dot(embeddings, query_emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
    )
    
    top_idx = scores.argsort()[::-1][:top_k]
    return [
        (scores[i], chunk_texts[i], chunks[i].get("metadata", {}))
        for i in top_idx
    ]


# ── Query Rewriting ──
def rewrite_query(user_query: str, chat_history: list[dict]) -> str:
    """Use LLM to rewrite query using chat history for better retrieval."""
    if not chat_history:
        return user_query

    history_str = "\n".join(
        f"{'User' if turn['role'] == 'user' else 'AI'}: {turn['content'][:200]}"
        for turn in chat_history[-6:]  # Last 3 exchanges
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


# ── Generate response with context ──
def ask(
    question: str,
    chat_history: list[dict],
    embeddings: np.ndarray,
    chunk_texts: list[str],
    chunks: list[dict]
) -> tuple[str, str, list]:
    """Ask a question and get an answer based on retrieved context."""
    
    # Step 1: Rewrite query for better retrieval
    rewritten = rewrite_query(question, chat_history)

    # Step 2: Retrieve with rewritten query
    results = retrieve(rewritten, embeddings, chunk_texts, chunks, top_k=3)
    context_str = "\n\n".join(
        f"[Similarity: {s:.3f}]\n{c}"
        for s, c, _ in results
    )

    # Step 3: Build prompt with history + context
    history_str = ""
    if chat_history:
        recent = chat_history[-6:]  # Last 3 exchanges
        history_str = "RECENT CONVERSATION:\n" + "\n".join(
            f"{'User' if t['role'] == 'user' else 'AI'}: {t['content'][:150]}..."
            for t in recent
        ) + "\n\n"

    prompt = f"""You are an MBSR (Mindfulness-Based Stress Reduction) expert 
answering questions about the MBSR Handbook and mindfulness practices.
Use the provided context from the knowledge base to give accurate, helpful answers.
If the context doesn't contain relevant information, say so and provide what general 
knowledge you have about the topic.

{history_str}RETRIEVED CONTEXT:
{context_str}

CURRENT QUESTION: {question}

ANSWER:"""

    response = model.generate_content(prompt)
    return response.text, rewritten, results


# ── CLI ──
def main():
    """Main conversational RAG loop."""
    print("=" * 70)
    print("EXAMPLE 8: Conversational RAG with Real MBSR Data")
    print("=" * 70)
    
    # Load chunks from JSONL
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "mbsr_chunks.jsonl"
    )
    
    chunks = load_chunks_from_jsonl(data_path)
    embeddings, chunk_texts = build_embedding_index(chunks)
    
    print("Commands:")
    print("  - Type your question to get an MBSR-based answer")
    print("  - Type 'history' to see chat memory")
    print("  - Type 'clear' to reset chat history")
    print("  - Type 'quit' to exit\n")

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
                    role = "User" if turn["role"] == "user" else "AI  "
                    content = turn["content"][:100] + "..." if len(turn["content"]) > 100 else turn["content"]
                    print(f"  {i}. {role}: {content}")
                print()
            continue

        if q.lower() == "clear":
            chat_history.clear()
            print("Chat history cleared.\n")
            continue

        # Get answer
        answer, rewritten, results = ask(
            q, chat_history, embeddings, chunk_texts, chunks
        )

        # Show rewriting if it changed
        if rewritten.lower() != q.lower():
            print(f"\n🔄 Rewritten query: \"{rewritten}\"")

        # Show retrieval info
        print(f"📎 Retrieved {len(results)} chunks")
        if results:
            print(f"   Top match similarity: {results[0][0]:.3f}")
            source = results[0][2].get("source_file", "unknown")
            page = results[0][2].get("page_number", "?")
            print(f"   Source: {source} (page {page})")

        print(f"\nAI: {answer}\n")

        # Store in history
        chat_history.append({"role": "user", "content": q})
        chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
