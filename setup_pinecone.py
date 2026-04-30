"""
Pinecone Setup Script for MBSR RAG
==================================
Creates a Pinecone vector database and indexes MBSR chunks using Pinecone's
serverless embedding feature. This eliminates the need for external embedding
APIs - Pinecone handles the embedding generation automatically.

USAGE:
  python3 setup_pinecone.py --create          # Create index and upload data
  python3 setup_pinecone.py --delete          # Delete index
  python3 setup_pinecone.py --list-indexes    # List all indexes
  python3 setup_pinecone.py --stats           # Show index statistics

REQUIREMENTS:
  - PINECONE_API_KEY in .env
  - PINECONE_ENVIRONMENT in .env (optional, defaults to us-east-1)

KEY BENEFITS OF THIS APPROACH:
  - Pinecone serverless embeddings: No need to call external APIs
  - Simplified workflow: Pass text directly, Pinecone handles embedding
  - Cost-effective: Uses Pinecone's efficient embedding service
  - Consistent: All embeddings use the same model and settings
"""
import os
import sys
import json
import argparse
import time
from typing import List, Dict
from dotenv import load_dotenv

from pinecone import Pinecone

load_dotenv(override=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

# Index configuration
INDEX_NAME = "mbsr-rag-index"
EMBEDDING_MODEL = "llama-text-embed-v2"  # Pinecone integrated embedding model
# DIMENSION and METRIC are determined automatically by create_index_for_model

# When uploading records, the field named EMBED_FIELD will be embedded by Pinecone.
# The field_map in create_index_for_model maps {"text": EMBED_FIELD}, where
# "text" is the model's expected input field name.
EMBED_FIELD = "chunk_text"  # Record field name that holds the text to embed
NAMESPACE = "mbsr"          # Pinecone namespace to organise vectors

# Data configuration
DATA_FILE = "data/mbsr_chunks.jsonl"
BATCH_SIZE = 90  # upsert_records batch size (keep under 100 for safety)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_config():
    """
    Validate that all required environment variables are set.
    
    This ensures we have:
    - PINECONE_API_KEY: Authentication for Pinecone API
    - PINECONE_ENVIRONMENT: Region where Pinecone index will be hosted
    """
    if not PINECONE_API_KEY:
        print("❌ PINECONE_API_KEY not found in .env")
        print("   Get your key at: https://www.pinecone.io/")
        sys.exit(1)
    if not PINECONE_ENVIRONMENT:
        print("❌ PINECONE_ENVIRONMENT not found in .env")
        print("   Use: us-east-1, us-west-2, eu-west-1, etc.")
        sys.exit(1)
    print("✅ Environment variables validated")


def init_pinecone():
    """
    Initialize and return a Pinecone client.
    
    This client will be used for all operations:
    - Creating/deleting indexes
    - Uploading/querying vectors
    - Managing embeddings
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc


def load_chunks_from_jsonl(filepath: str) -> List[Dict]:
    """
    Load MBSR chunks from a JSONL file (JSON Lines format).
    
    Each line contains a JSON object with:
    - id: Unique identifier for the chunk
    - text: The actual content to be embedded
    - metadata: Additional info (source file, page number, chunk index)
    
    Args:
        filepath: Path to the JSONL file
        
    Returns:
        List of dictionaries containing chunk data
    """
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
        print(f"✓ Loaded {len(chunks)} chunks from {filepath}")
        return chunks
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        sys.exit(1)


# ============================================================================
# INDEX MANAGEMENT FUNCTIONS
# ============================================================================

def create_index(pc: Pinecone):
    """
    Create a Pinecone integrated-embedding index using create_index_for_model.

    WHY create_index_for_model instead of create_index?
    - create_index_for_model creates an "integrated embedding" index.
    - The index knows which model to use and which record field to embed.
    - You can then call index.upsert_records() with plain text dicts and
      Pinecone embeds them automatically server-side.
    - create_index (standard) requires you to supply numeric float vectors;
      it does NOT accept raw text in upsert().

    field_map tells Pinecone: "embed the value in the record field 'chunk_text'
    using the model's input slot called 'text'".
    """
    try:
        # Check if index already exists
        existing_indexes = [idx.name for idx in pc.list_indexes()]

        if INDEX_NAME in existing_indexes:
            print(f"✓ Index '{INDEX_NAME}' already exists")
            return

        print(f"Creating integrated-embedding index '{INDEX_NAME}'...")
        pc.create_index_for_model(
            name=INDEX_NAME,
            cloud="aws",
            region=PINECONE_ENVIRONMENT,
            embed={
                "model": EMBEDDING_MODEL,
                "field_map": {"text": EMBED_FIELD}  # embed the 'chunk_text' field
            }
        )
        print(f"✓ Index '{INDEX_NAME}' created successfully")

        # Wait for index to be ready
        print("Waiting for index to be ready...")
        time.sleep(5)

    except Exception as e:
        print(f"❌ Error creating index: {e}")
        sys.exit(1)


def upload_chunks_to_pinecone(pc: Pinecone, chunks: List[Dict]):
    """
    Upload MBSR chunks to Pinecone using upsert_records (integrated embedding).

    Because the index was created with create_index_for_model, we use
    index.upsert_records() instead of index.upsert().

    Record format expected by upsert_records:
    {
        "_id": "unique-id",          # required: record identifier
        "chunk_text": "...",          # the field Pinecone will embed (matches field_map)
        "source": "...",              # metadata fields (any extra keys are stored as metadata)
        "page": 1,
        "chunk_idx": 0
    }

    Pinecone reads chunk_text, generates the embedding server-side, and stores
    both the vector and all remaining fields as metadata.

    Args:
        pc: Pinecone client
        chunks: List of chunk dicts from load_chunks_from_jsonl()
    """
    try:
        index = pc.Index(INDEX_NAME)

        print(f"\nPreparing {len(chunks)} records for upload...")
        print(f"Note: Pinecone will embed the '{EMBED_FIELD}' field automatically\n")

        print(f"Uploading to Pinecone in batches of {BATCH_SIZE}...")
        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

        for i in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1

            # Build records as plain dicts.
            # _id   → Pinecone record identifier
            # chunk_text → the field that gets embedded (set in field_map)
            # other fields → stored as metadata for retrieval
            records = [
                {
                    "_id": chunk["id"],
                    EMBED_FIELD: chunk["text"],          # field Pinecone embeds
                    "source": chunk["metadata"].get("source_file", "unknown"),
                    "page": chunk["metadata"].get("page_number", 0),
                    "chunk_idx": chunk["metadata"].get("chunk_index", 0)
                }
                for chunk in batch_chunks
            ]

            # upsert_records handles embedding + storage in one call
            index.upsert_records(NAMESPACE, records)
            print(f"  ✓ Batch {batch_num}/{total_batches} uploaded ({len(records)} records)")

            if batch_num < total_batches:
                time.sleep(1)

        print(f"\n✓ All {len(chunks)} records uploaded successfully")

        # Show final index stats
        time.sleep(2)
        stats = index.describe_index_stats()
        print(f"\n📊 Index Statistics:")
        print(f"   Total vectors: {stats.total_vector_count}")
        print(f"   Namespace: '{NAMESPACE}'")
        print(f"   Ready for queries: {'✓' if stats.total_vector_count > 0 else '✗'}")

    except Exception as e:
        print(f"❌ Error uploading to Pinecone: {e}")
        sys.exit(1)


# ============================================================================
# INDEX UTILITY FUNCTIONS
# ============================================================================

def delete_index(pc: Pinecone):
    """
    Delete a Pinecone index after confirming with the user.
    
    This removes:
    - All vectors and embeddings
    - All metadata and search filters
    - The entire index resource
    
    WARNING: This action cannot be undone!
    """
    try:
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        
        if INDEX_NAME not in existing_indexes:
            print(f"❌ Index '{INDEX_NAME}' does not exist")
            return
        
        # Confirm with user
        response = input(f"⚠️  Are you sure you want to delete '{INDEX_NAME}'? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled.")
            return
        
        # Delete the index
        pc.delete_index(INDEX_NAME)
        print(f"✓ Index '{INDEX_NAME}' deleted successfully")
        
    except Exception as e:
        print(f"❌ Error deleting index: {e}")
        sys.exit(1)


def list_indexes(pc: Pinecone):
    """
    List all available Pinecone indexes in your account.
    
    Shows:
    - Index name
    - Vector dimension
    - Status and creation date
    """
    try:
        indexes = pc.list_indexes()
        if not indexes:
            print("No indexes found")
            return
        
        print("\nAvailable Pinecone indexes:")
        for idx in indexes:
            marker = "→" if idx.name == INDEX_NAME else " "
            print(f"  {marker} {idx.name} (dim: {idx.dimension})")
        
    except Exception as e:
        print(f"❌ Error listing indexes: {e}")
        sys.exit(1)


def show_stats(pc: Pinecone):
    """
    Display detailed statistics about the MBSR RAG index.
    
    Shows:
    - Total number of vectors (chunks)
    - Vector dimension
    - Index namespaces
    - Ready status
    """
    try:
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        
        if INDEX_NAME not in existing_indexes:
            print(f"❌ Index '{INDEX_NAME}' does not exist")
            return
        
        # Get index reference and stats
        index = pc.Index(INDEX_NAME)
        stats = index.describe_index_stats()
        
        print(f"\n📊 Index Details: {INDEX_NAME}")
        print(f"   Total vectors: {stats.total_vector_count}")
        print(f"   Namespace: '{NAMESPACE}'")
        print(f"   Embedding Model: {EMBEDDING_MODEL}")
        print(f"   Status: {'Ready for queries ✓' if stats.total_vector_count > 0 else 'Empty'}")
        
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        sys.exit(1)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point with command-line argument parsing.
    
    Supports operations:
    - --create: Create index and upload all MBSR chunks
    - --delete: Delete the index
    - --list-indexes: Show all available indexes
    - --stats: Display current index statistics
    """
    parser = argparse.ArgumentParser(
        description="Setup Pinecone index for MBSR RAG with serverless embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 setup_pinecone.py --create          Create index and upload data
  python3 setup_pinecone.py --stats           Show index statistics
  python3 setup_pinecone.py --list-indexes    List all indexes
  python3 setup_pinecone.py --delete          Delete index (requires confirmation)
        """
    )
    
    parser.add_argument("--create", action="store_true", 
                       help="Create index and upload MBSR chunks")
    parser.add_argument("--delete", action="store_true", 
                       help="Delete index (requires confirmation)")
    parser.add_argument("--list-indexes", action="store_true", 
                       help="List all available indexes")
    parser.add_argument("--stats", action="store_true", 
                       help="Show index statistics")
    
    args = parser.parse_args()
    
    # Display header
    print("=" * 70)
    print("PINECONE SETUP FOR MBSR RAG (with Serverless Embeddings)")
    print("=" * 70)
    
    # Validate configuration
    validate_config()
    
    # Initialize Pinecone client
    pc = init_pinecone()
    
    # Execute requested operation
    if args.create:
        print(f"\n🔄 Creating Pinecone index and uploading MBSR data...\n")
        chunks = load_chunks_from_jsonl(DATA_FILE)
        create_index(pc)
        upload_chunks_to_pinecone(pc, chunks)
        print("\n✅ Setup complete! Your index is ready for RAG queries.")
    
    elif args.delete:
        print(f"\n🔄 Deleting index...\n")
        delete_index(pc)
    
    elif args.list_indexes:
        print(f"\n🔄 Listing indexes...\n")
        list_indexes(pc)
    
    elif args.stats:
        print(f"\n🔄 Getting index statistics...\n")
        show_stats(pc)
    
    else:
        # Show help if no arguments
        parser.print_help()


if __name__ == "__main__":
    main()
