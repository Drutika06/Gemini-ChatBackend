"""
Example 6: PDF RAG Pipeline — End-to-End Document Q&A
======================================================
CONCEPT: Full RAG pipeline on a REAL PDF document:
  1. LOAD    — Extract text from PDF using PyMuPDF
  2. CHUNK   — Split into overlapping sentence-based chunks
  3. EMBED   — Create vector embeddings for each chunk
  4. STORE   — Keep embeddings in memory (numpy array = simple vector store)
  5. RETRIEVE — Semantic search with cosine similarity
  6. GENERATE — LLM answers grounded in retrieved context

NEW IDEA: This is a production-grade RAG pattern! The only difference from
          real systems (Pinecone, ChromaDB, etc.) is the vector store — we
          use a numpy array instead of a database. The pattern is identical.

USES: Your actual MBSR PDF from knowledge-sources/

RUN:  python3 examples/06_pdf_rag.py
"""
import os, sys, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(override=True)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")

# ── STEP 1: Load PDF ──
PDF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge-sources", "MBSR-Handbook-Single-Page-Final.pdf")

def extract_pdf_text(path: str) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    import fitz  # PyMuPDF
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text

print("📄 Loading PDF...")
raw_text = extract_pdf_text(PDF_PATH)
print(f"   Extracted {len(raw_text)} characters from {os.path.basename(PDF_PATH)}")

# ── STEP 2: Chunk — sentence-based with overlap ──
def chunk_text(text: str, max_sentences: int = 5, overlap: int = 1) -> list[str]:
    """Split text into chunks of N sentences with overlap."""
    # Clean up whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]  # skip tiny fragments
    
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = " ".join(sentences[i:i + max_sentences])
        if len(chunk) > 50:  # skip trivially small chunks
            chunks.append(chunk)
        i += max_sentences - overlap
    return chunks

chunks = chunk_text(raw_text, max_sentences=5, overlap=1)
print(f"   Split into {len(chunks)} chunks (5 sentences each, 1 overlap)")

# ── STEP 3: Embed ──
def get_embedding(text: str, task: str = "retrieval_document") -> np.ndarray:
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type=task
    )
    return np.array(result['embedding'])

print("🧠 Building embedding index (this may take a moment)...")
chunk_embeddings = []
BATCH_SIZE = 5
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i + BATCH_SIZE]
    for chunk in batch:
        chunk_embeddings.append(get_embedding(chunk))
    done = min(i + BATCH_SIZE, len(chunks))
    print(f"   Embedded {done}/{len(chunks)} chunks...")

chunk_embeddings = np.array(chunk_embeddings)
print(f"   Index ready! Shape: {chunk_embeddings.shape}")

# ── STEP 4 & 5: Retrieve ──
def retrieve(query: str, top_k: int = 4) -> list[tuple[float, str]]:
    query_emb = get_embedding(query, task="retrieval_query")
    # Cosine similarity
    scores = np.dot(chunk_embeddings, query_emb) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_emb)
    )
    top_indices = scores.argsort()[::-1][:top_k]
    return [(scores[i], chunks[i]) for i in top_indices]

# ── STEP 6: Generate ──
def ask(question: str) -> str:
    results = retrieve(question)
    context_str = "\n\n".join(f"[Relevance: {s:.3f}]\n{c}" for s, c in results)

    prompt = f"""You are answering questions about the MBSR (Mindfulness-Based Stress Reduction) Handbook.
Use ONLY the provided context to answer. Quote specific details when possible. If the context doesn't contain the answer, say so.

CONTEXT FROM PDF:
{context_str}

QUESTION: {question}

ANSWER:"""
    return model.generate_content(prompt).text

# ── CLI ──
print("\n" + "=" * 60)
print("EXAMPLE 6: PDF RAG Pipeline")
print("=" * 60)
print(f"Source: {os.path.basename(PDF_PATH)}")
print(f"Index: {len(chunks)} chunks with Gemini embeddings")
print("\nAsk anything about the MBSR Handbook!")
print("Type 'sources' after a question to see the raw retrieved chunks.")
print("Type 'quit' to exit.\n")

last_results = []

while True:
    try:
        q = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!"); break
    if not q:
        continue
    if q.lower() in ("quit", "exit", "q"):
        break

    if q.lower() == "sources" and last_results:
        print("\n── Retrieved sources ──")
        for i, (score, chunk) in enumerate(last_results):
            print(f"\n[{i+1}] Relevance: {score:.3f}")
            print(f"    {chunk}")
        print()
        continue

    last_results = retrieve(q)
    print(f"\n📎 Retrieved {len(last_results)} chunks (top score: {last_results[0][0]:.3f})")
    answer = ask(q)
    print(f"\nAI: {answer}")
    print("(Type 'sources' to see the retrieved chunks)\n")
