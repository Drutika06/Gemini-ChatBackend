# MinerU Document Extraction Guide

This guide explains how to use **MinerU** (Mine Your URl/document Universe) to extract structured data from documents (PDFs, images, scanned documents) for your RAG (Retrieval-Augmented Generation) pipeline.

## Quick Start: Using `06b_pdf_to_jsonl.py`

Your project already includes a ready-to-use script for converting PDFs to embedding-ready JSONL chunks using MinerU.

### Basic Usage

```bash
# Process default PDF (MBSR-Handbook-Single-Page-Final.pdf)
python3 examples/06b_pdf_to_jsonl.py

# Specify custom output location
python3 examples/06b_pdf_to_jsonl.py --output data/custom_chunks.jsonl

# Adjust chunking parameters
python3 examples/06b_pdf_to_jsonl.py \
    --output data/mbsr_chunks.jsonl \
    --max-sentences 6 \
    --overlap 2
```

### Output Format

The script generates JSONL with this structure:

```json
{
  "id": "mbsr-p0001-c000001",
  "text": "chunk text ...",
  "metadata": {
    "source_file": "MBSR-Handbook-Single-Page-Final.pdf",
    "page_number": 1,
    "chunk_index": 1,
    "char_count": 512,
    "token_estimate": 128
  }
}
```

### Available Options

```bash
python3 examples/06b_pdf_to_jsonl.py --help
```

Key parameters:
- `--input` : PDF file path (default: MBSR-Handbook-Single-Page-Final.pdf)
- `--output` : Output JSONL file (default: data/mbsr_chunks.jsonl)
- `--max-sentences` : Sentences per chunk (default: 5)
- `--overlap` : Sentence overlap between chunks (default: 1)
- `--min-chars` : Minimum chunk size in characters
- `--backend` : MinerU backend (default: auto-detect)
- `--method` : Extraction method (default: auto)
- `--lang` : OCR language code (e.g., en, zh, hi)
- `--start-page` : Start page number (default: 1)
- `--end-page` : End page number (default: all)
- `--no-formula` : Disable formula extraction
- `--no-table` : Disable table extraction

### Processing Multiple PDFs

Batch process all PDFs in a directory:

```bash
# Process all PDFs in knowledge-sources/
for pdf in knowledge-sources/*.pdf; do
    python3 examples/06b_pdf_to_jsonl.py \
        --input "$pdf" \
        --output "data/$(basename "$pdf" .pdf)_chunks.jsonl"
done
```

### Integration with Your RAG Pipeline

Once JSONL files are created, use them with your embedding pipeline:

```python
# In your embedding/RAG script
import json

def load_chunks(jsonl_file):
    """Load chunks from JSONL file."""
    chunks = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

# Use with embeddings
chunks = load_chunks('data/mbsr_chunks.jsonl')

for chunk in chunks:
    # Generate embeddings
    embedding = embed_function(chunk['text'])
    # Upsert to vector DB (Pinecone, Weaviate, etc.)
    vector_db.upsert(
        id=chunk['id'],
        embedding=embedding,
        metadata=chunk['metadata']
    )
```

---



## What is MinerU?

MinerU is an advanced document parsing and extraction library that:
- **Extracts text, tables, and images** from PDFs and scanned documents
- **Preserves document structure** and layout information
- **Handles complex layouts** (multi-column, tables, figures)
- **Converts documents to markdown** for better readability
- **Supports batch processing** of multiple documents
- **Works with various formats**: PDF, images (PNG, JPG), etc.

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Step 1: Install MinerU

```bash
# Via pip
pip install mineru

# Or via conda
conda install -c conda-forge mineru
```

### Step 2: Install Optional Dependencies

For PDF processing and OCR support:

```bash
# For better PDF processing
pip install pymupdf4llm

# For OCR (Optical Character Recognition) - handles scanned documents
pip install paddleocr paddlepaddle
```

### Step 3: Add to Your Requirements

Add to `requirements.txt`:

```
mineru>=0.2.0
pymupdf4llm>=0.1.0
paddleocr>=2.7.0.0
paddlepaddle>=2.5.0
```

Then install:

```bash
pip install -r requirements.txt
```

## Basic Usage

### Simple Document Extraction

```python
from mineru import DocumentProcessor

# Initialize processor
processor = DocumentProcessor()

# Extract from a single PDF
result = processor.process_document('path/to/document.pdf')

# Access extracted content
text = result.text          # Full extracted text
tables = result.tables      # Extracted tables
images = result.images      # Extracted images
metadata = result.metadata  # Document metadata
```

### Save as Markdown

```python
from mineru import DocumentProcessor

processor = DocumentProcessor()
result = processor.process_document('document.pdf')

# Save as markdown (preserves structure)
with open('document.md', 'w') as f:
    f.write(result.to_markdown())
```

## Integration with Your RAG Pipeline

### 1. Extract and Convert to JSONL

Create a script `mineru_to_jsonl.py`:

```python
import json
from mineru import DocumentProcessor
from pathlib import Path
from typing import List, Dict

def extract_and_chunk_documents(
    pdf_path: str,
    output_jsonl: str,
    chunk_size: int = 1000,
    overlap: int = 100
) -> None:
    """
    Extract documents using MinerU and save as JSONL chunks.
    
    Args:
        pdf_path: Path to PDF file or directory of PDFs
        output_jsonl: Output JSONL file path
        chunk_size: Characters per chunk
        overlap: Character overlap between chunks
    """
    
    processor = DocumentProcessor()
    documents = []
    
    # Handle single file or directory
    if Path(pdf_path).is_dir():
        pdf_files = list(Path(pdf_path).glob('*.pdf'))
    else:
        pdf_files = [Path(pdf_path)]
    
    # Process each PDF
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file}")
        
        try:
            result = processor.process_document(str(pdf_file))
            
            # Extract text with structure
            text = result.to_markdown()
            metadata = {
                'source': str(pdf_file),
                'title': pdf_file.stem,
                'tables_count': len(result.tables),
                'images_count': len(result.images)
            }
            
            documents.append({
                'text': text,
                'metadata': metadata
            })
            
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            continue
    
    # Chunk documents
    chunks = []
    for doc in documents:
        text = doc['text']
        metadata = doc['metadata']
        
        # Simple chunking strategy
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            
            chunks.append({
                'text': chunk_text,
                'source': metadata['source'],
                'title': metadata['title'],
                'position': i
            })
    
    # Save to JSONL
    with open(output_jsonl, 'w') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')
    
    print(f"Saved {len(chunks)} chunks to {output_jsonl}")

# Usage
if __name__ == '__main__':
    extract_and_chunk_documents(
        pdf_path='knowledge-sources/',
        output_jsonl='data/mineru_chunks.jsonl',
        chunk_size=1000,
        overlap=100
    )
```

### 2. Batch Processing Multiple Documents

```python
from mineru import DocumentProcessor
from pathlib import Path
import json
from datetime import datetime

def batch_process_documents(input_dir: str, output_dir: str) -> Dict:
    """Batch process all PDFs in a directory."""
    
    processor = DocumentProcessor()
    results = {
        'processed': 0,
        'failed': 0,
        'documents': []
    }
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for pdf_file in input_path.glob('*.pdf'):
        try:
            print(f"Processing: {pdf_file.name}")
            result = processor.process_document(str(pdf_file))
            
            # Save markdown
            md_file = output_path / f"{pdf_file.stem}.md"
            with open(md_file, 'w') as f:
                f.write(result.to_markdown())
            
            # Save metadata
            meta_file = output_path / f"{pdf_file.stem}_metadata.json"
            with open(meta_file, 'w') as f:
                json.dump({
                    'source': str(pdf_file),
                    'processed_at': datetime.now().isoformat(),
                    'tables': len(result.tables),
                    'images': len(result.images),
                    'pages': result.metadata.get('pages', 'unknown')
                }, f, indent=2)
            
            results['documents'].append({
                'name': pdf_file.name,
                'status': 'success',
                'output': str(md_file)
            })
            results['processed'] += 1
            
        except Exception as e:
            results['documents'].append({
                'name': pdf_file.name,
                'status': 'failed',
                'error': str(e)
            })
            results['failed'] += 1
            print(f"Failed to process {pdf_file.name}: {e}")
    
    # Save summary
    summary_file = output_path / 'processing_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

# Usage
if __name__ == '__main__':
    results = batch_process_documents(
        input_dir='knowledge-sources/',
        output_dir='knowledge-sources/extracted/'
    )
    print(f"\nProcessing complete:")
    print(f"  Successful: {results['processed']}")
    print(f"  Failed: {results['failed']}")
```

### 3. Extract Tables for Structured Data

```python
from mineru import DocumentProcessor
import pandas as pd

def extract_tables_from_document(pdf_path: str, output_dir: str = '.') -> List[pd.DataFrame]:
    """Extract and save tables from PDF."""
    
    processor = DocumentProcessor()
    result = processor.process_document(pdf_path)
    
    tables = []
    for idx, table in enumerate(result.tables):
        # Convert to pandas DataFrame
        df = pd.DataFrame(table.data)
        
        # Save as CSV
        csv_path = f"{output_dir}/table_{idx}.csv"
        df.to_csv(csv_path, index=False)
        
        tables.append(df)
        print(f"Saved table {idx} to {csv_path}")
    
    return tables

# Usage
if __name__ == '__main__':
    tables = extract_tables_from_document(
        'knowledge-sources/example.pdf',
        'extracted_tables/'
    )
```

## Advanced Configuration

### Custom Processing Parameters

```python
from mineru import DocumentProcessor

# Configure processor with custom settings
config = {
    'extract_text': True,
    'extract_tables': True,
    'extract_images': True,
    'use_ocr': True,           # Enable OCR for scanned documents
    'language': 'en',          # Language for OCR
    'preserve_layout': True,   # Maintain document layout
    'output_format': 'markdown' # markdown, text, json
}

processor = DocumentProcessor(**config)
result = processor.process_document('document.pdf')
```

### Handling Scanned Documents

For PDFs that are scanned images:

```python
from mineru import DocumentProcessor

# Enable OCR for scanned documents
config = {
    'use_ocr': True,
    'ocr_language': 'en+hi',  # English + Hindi (multilingual)
}

processor = DocumentProcessor(**config)
result = processor.process_document('scanned_document.pdf')

# Save extracted text
with open('extracted_text.txt', 'w') as f:
    f.write(result.text)
```

## Integration with Your Existing RAG System

### Update Your RAG Pipeline

Modify `examples/06b_pdf_to_jsonl.py` to use MinerU:

```python
from mineru import DocumentProcessor
from pathlib import Path
import json

def process_with_mineru(pdf_path: str, output_jsonl: str):
    """Process PDFs using MinerU instead of PyPDF2."""
    
    processor = DocumentProcessor()
    result = processor.process_document(pdf_path)
    
    # Get markdown output (better structure)
    text = result.to_markdown()
    
    # Chunk and save
    chunks = []
    chunk_size = 1000
    
    for i in range(0, len(text), chunk_size - 100):
        chunk = {
            'text': text[i:i+chunk_size],
            'source': str(pdf_path),
            'chunk_id': len(chunks)
        }
        chunks.append(chunk)
    
    with open(output_jsonl, 'w') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')
    
    return len(chunks)
```

## Troubleshooting

### Issue: "MinerU not found" error
```bash
# Ensure proper installation
pip install --upgrade mineru
```

### Issue: OCR not working
```bash
# Install OCR dependencies
pip install paddleocr paddlepaddle
```

### Issue: Memory error with large documents
```python
# Process document in chunks
processor = DocumentProcessor(chunk_pages=10)  # Process 10 pages at a time
result = processor.process_document('large_document.pdf')
```

### Issue: Slow processing
```python
# Enable parallel processing
processor = DocumentProcessor(num_workers=4)  # Use 4 parallel workers
```

## Performance Tips

1. **Batch Processing**: Process multiple documents together
2. **Disable Unused Features**: Only enable image/table extraction if needed
3. **Adjust Chunk Size**: Larger chunks process faster but use more memory
4. **Use Caching**: Cache processing results for repeated operations

```python
from mineru import DocumentProcessor
import pickle

# Cache results
def cached_process(pdf_path, cache_file='cache.pkl'):
    try:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    except:
        processor = DocumentProcessor()
        result = processor.process_document(pdf_path)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        return result
```

## Comparison with Alternative Tools

| Feature | MinerU | PyPDF2 | pdfplumber |
|---------|--------|--------|-----------|
| Text Extraction | ✓ | ✓ | ✓ |
| Table Extraction | ✓ | ✗ | ✓ |
| Image Extraction | ✓ | ✗ | ✓ |
| OCR Support | ✓ | ✗ | ✗ |
| Layout Preservation | ✓ | ✗ | ✓ |
| Scanned PDFs | ✓ | ✗ | ✗ |
| Markdown Output | ✓ | ✗ | ✗ |

## Resources

- [MinerU GitHub Repository](https://github.com/opendatalab/mineru)
- [MinerU Documentation](https://mineru.readthedocs.io)
- [OpenDataLab MinerU](https://huggingface.co/opendatalab/MinerU)

## Next Steps

1. Install MinerU and test with a sample PDF
2. Integrate into your RAG pipeline for better document processing
3. Experiment with OCR for scanned documents
4. Batch process your knowledge sources
5. Compare extraction quality with your current method

---

**Last Updated**: April 2026  
**Compatible with**: MinerU 0.2.0+, Python 3.8+
