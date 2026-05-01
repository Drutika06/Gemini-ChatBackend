"""
Example 6b: Convert PDF to Embedding-Ready JSONL (MinerU)
==========================================================
Creates chunked JSONL records from a PDF using MinerU so they can be
embedded locally (or by any embedding pipeline) and then upserted into Pinecone.

Default input:
  knowledge-sources/MBSR-Handbook-Single-Page-Final.pdf

Output format (one JSON object per line):
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

Run:
  python3 examples/06b_pdf_to_jsonl.py
  python3 examples/06b_pdf_to_jsonl.py --output data/mbsr_chunks.jsonl --max-sentences 6 --overlap 2
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_PDF = Path(__file__).resolve().parent.parent / "knowledge-sources" / "MBSR-Handbook-Single-Page-Final.pdf"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "mbsr_chunks.jsonl"


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    # Lightweight sentence splitter that works well for general PDF text.
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in pieces if len(p.strip()) > 20]


def sentence_chunks(sentences: List[str], max_sentences: int, overlap: int, min_chars: int) -> List[str]:
    if not sentences:
        return []

    if max_sentences < 1:
        raise ValueError("max_sentences must be >= 1")

    if overlap >= max_sentences:
        raise ValueError("overlap must be smaller than max_sentences")

    step = max_sentences - overlap
    chunks: List[str] = []

    i = 0
    while i < len(sentences):
        chunk = " ".join(sentences[i : i + max_sentences])
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        i += step

    return chunks


def _collect_text(block: dict) -> str:
    parts: List[str] = []
    for key in ("text", "code_body", "latex", "html", "table_body"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
        elif isinstance(value, list):
            pieces = [v for v in value if isinstance(v, str) and v.strip()]
            if pieces:
                parts.append(" ".join(pieces))
    return normalize_whitespace(" ".join(parts))


def _read_content_list_json(content_list_path: Path) -> List[Tuple[int, str]]:
    with content_list_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return []

    by_page: Dict[int, List[str]] = {}
    for block in data:
        if not isinstance(block, dict):
            continue
        page_idx = block.get("page_idx")
        if not isinstance(page_idx, int):
            continue
        text = _collect_text(block)
        if not text:
            continue
        by_page.setdefault(page_idx + 1, []).append(text)

    pages: List[Tuple[int, str]] = []
    for page_number in sorted(by_page):
        page_text = normalize_whitespace(" ".join(by_page[page_number]))
        if page_text:
            pages.append((page_number, page_text))
    return pages


def extract_pages_with_mineru(
    pdf_path: Path,
    backend: str,
    method: str,
    lang: str | None,
    start_page: int,
    end_page: int | None,
    formula: bool,
    table: bool,
) -> List[Tuple[int, str]]:
    search_paths: List[str] = []
    search_paths.append(str(Path(sys.executable).parent))
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        search_paths.append(str(Path(virtual_env) / "bin"))

    mineru_bin = shutil.which("mineru")
    if mineru_bin is None:
        for p in search_paths:
            mineru_bin = shutil.which("mineru", path=p)
            if mineru_bin:
                break

    if mineru_bin is None:
        for p in search_paths:
            candidate = Path(p) / "mineru"
            if candidate.exists() and candidate.is_file():
                mineru_bin = str(candidate)
                break

    if mineru_bin is None:
        raise RuntimeError(
            f"MinerU CLI not found. Install it in this interpreter env: {sys.executable}"
        )

    with tempfile.TemporaryDirectory(prefix="mineru_extract_") as temp_dir:
        out_dir = Path(temp_dir)
        cmd = [
            mineru_bin,
            "-p",
            str(pdf_path),
            "-o",
            str(out_dir),
            "-b",
            backend,
            "-m",
            method,
            "-s",
            str(start_page),
            "-f",
            "true" if formula else "false",
            "-t",
            "true" if table else "false",
        ]
        if end_page is not None:
            cmd.extend(["-e", str(end_page)])
        if lang:
            cmd.extend(["-l", lang])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            details = stderr or stdout or "No output from mineru command"
            raise RuntimeError(f"MinerU parsing failed: {details}")

        content_list_files = sorted(out_dir.rglob("*_content_list.json"))
        if content_list_files:
            pages = _read_content_list_json(content_list_files[0])
            if pages:
                return pages

        # Fallback to markdown if content_list is unavailable.
        md_files = sorted(out_dir.rglob("*.md"))
        if not md_files:
            return []
        md_text = normalize_whitespace(md_files[0].read_text(encoding="utf-8", errors="ignore"))
        return [(1, md_text)] if md_text else []


def iter_records(
    pages: Iterable[Tuple[int, str]],
    source_file: str,
    max_sentences: int,
    overlap: int,
    min_chars: int,
    id_prefix: str,
):
    chunk_counter = 0
    for page_number, page_text in pages:
        sentences = split_sentences(page_text)
        chunks = sentence_chunks(sentences, max_sentences=max_sentences, overlap=overlap, min_chars=min_chars)
        for chunk in chunks:
            chunk_counter += 1
            record_id = f"{id_prefix}-p{page_number:04d}-c{chunk_counter:06d}"
            yield {
                "id": record_id,
                "text": chunk,
                "metadata": {
                    "source_file": source_file,
                    "page_number": page_number,
                    "chunk_index": chunk_counter,
                    "char_count": len(chunk),
                    # Simple approximation; replace with tokenizer if needed.
                    "token_estimate": max(1, len(chunk) // 4),
                },
            }


def write_jsonl(records: Iterable[dict], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a PDF into embedding-ready JSONL chunks.")
    parser.add_argument("--input", type=Path, default=DEFAULT_PDF, help="Path to input PDF")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to output JSONL")
    parser.add_argument(
        "--backend",
        type=str,
        default="pipeline",
        help="MinerU backend (e.g. hybrid-auto-engine, pipeline, vlm-auto-engine)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="txt",
        choices=["auto", "txt", "ocr"],
        help="MinerU parse method. txt is usually best for digital PDFs.",
    )
    parser.add_argument("--lang", type=str, default="en", help="Optional MinerU OCR language code (e.g. en, ch)")
    parser.add_argument("--start-page", type=int, default=0, help="Start page for MinerU parsing (0-based)")
    parser.add_argument("--end-page", type=int, default=None, help="End page for MinerU parsing (0-based, inclusive)")
    parser.add_argument("--formula", action="store_true", help="Enable formula parsing in MinerU")
    parser.add_argument("--table", action="store_true", help="Enable table parsing in MinerU")
    parser.add_argument("--max-sentences", type=int, default=5, help="Sentences per chunk")
    parser.add_argument("--overlap", type=int, default=1, help="Overlapping sentences between consecutive chunks")
    parser.add_argument("--min-chars", type=int, default=80, help="Minimum characters for keeping a chunk")
    parser.add_argument(
        "--id-prefix",
        type=str,
        default="mbsr",
        help="Prefix used for chunk IDs (useful for Pinecone namespace hygiene)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input PDF not found: {args.input}")

    pages = extract_pages_with_mineru(
        pdf_path=args.input,
        backend=args.backend,
        method=args.method,
        lang=args.lang,
        start_page=args.start_page,
        end_page=args.end_page,
        formula=args.formula,
        table=args.table,
    )
    records = iter_records(
        pages=pages,
        source_file=args.input.name,
        max_sentences=args.max_sentences,
        overlap=args.overlap,
        min_chars=args.min_chars,
        id_prefix=args.id_prefix,
    )
    written = write_jsonl(records, args.output)

    print(f"Input PDF: {args.input}")
    print(f"MinerU backend: {args.backend}")
    print(f"MinerU method: {args.method}")
    print(f"Pages with text: {len(pages)}")
    print(f"Output JSONL: {args.output}")
    print(f"Chunks written: {written}")
    print("Done. This JSONL is ready for local embedding or Pinecone ingestion pipelines.")


if __name__ == "__main__":
    main()
