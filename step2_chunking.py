#!/usr/bin/env python3
# step2_chunking.py
"""
Step 2: Chunk extracted text into overlapping chunks and save metadata.

This script takes a plain text file (output from the PDF extraction step) and
splits it into word-based chunks with a configurable overlap. Each chunk is
saved as its own text file under a chunks directory, and a JSON metadata file
is produced containing the mapping between chunk ids and their files.

Why chunking?
- Retrieval models perform better with moderately sized chunks and some overlap
  to preserve context across boundaries.

Usage:
    python step2_chunking.py --input docs/hr_policy.txt --out metadata.json --chunks_dir chunks \
        --chunk_size 120 --overlap 30 --preview 5
"""
import re
import json
from pathlib import Path
import argparse

def clean_text(text: str) -> str:
    """
    Normalize artifacts from PDF extraction and lightly clean the text.

    Operations performed:
    - Replace common PDF artifact tokens like "(cid:127)" with a bullet
    - Remove lines that are only numbers (often table/chart remnants)
    - Squash runs of 3+ newlines down to exactly 2 (paragraph boundaries)
    - Trim trailing whitespace at end of each line
    """
    # Replace common PDF artifacts such as (cid:127) with a bullet or whitespace
    text = re.sub(r"\(cid:\d+\)", "•", text)
    # Remove isolated lines that are just numbers (chart extraction artifacts)
    text = re.sub(r"(?m)^[ \t]*\d+[ \t]*\n+", "", text)
    # Normalize multiple newlines to two newlines (paragraph separation)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Trim whitespace on each line
    text = "\n".join([ln.rstrip() for ln in text.splitlines()])
    return text.strip()

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50):
    """
    Split text into word-based chunks with overlap.

    Args:
        text: The full input string to chunk.
        chunk_size: Target number of words per chunk.
        overlap: Number of words to repeat from the tail of the previous
                 chunk at the head of the next chunk. Helps preserve context.

    Returns:
        A list of chunk strings.

    Notes:
        - Ensures overlap < chunk_size to avoid infinite loops.
        - Returns an empty list when the input has no words.
    """
    words = text.split()
    if not words:
        return []

    # Guard against invalid settings that could cause non-advancing windows
    effective_overlap = min(overlap, max(0, chunk_size - 1))

    chunks = []
    start = 0
    n = len(words)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == n:
            break
        # Move the start forward but keep some overlap from the previous chunk
        start = end - effective_overlap
    return chunks

def main(args):
    # Resolve and validate paths
    inp = Path(args.input)
    assert inp.exists(), f"Input file not found: {inp}"
    out_meta = Path(args.out)
    chunks_dir = Path(args.chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Ensure sane chunking parameters (avoid overlap >= chunk_size)
    if args.overlap >= args.chunk_size and args.chunk_size > 0:
        print(f"[warn] overlap ({args.overlap}) >= chunk_size ({args.chunk_size}). Adjusting overlap to {args.chunk_size - 1}.")
        args.overlap = args.chunk_size - 1

    # 1) Load raw text
    raw = inp.read_text(encoding="utf-8")

    # 2) Clean text to remove PDF artifacts and normalize newlines
    cleaned = clean_text(raw)

    # 3) Chunk the cleaned text into overlapping windows
    #    Consider splitting by page markers first if desired (not enabled by default).
    chunks = chunk_text(cleaned, chunk_size=args.chunk_size, overlap=args.overlap)

    # 4) Persist chunks and build metadata for downstream indexing
    metadata = []
    for i, ch in enumerate(chunks):
        meta = {
            "doc": inp.name,
            "chunk_id": i,
            "word_count": len(ch.split()),
            "text_file": f"{chunks_dir.name}/{i}.txt",
        }
        metadata.append(meta)
        (chunks_dir / f"{i}.txt").write_text(ch, encoding="utf-8")

    # 5) Write metadata json
    out_meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    # 6) Print summary / preview to help quick inspection during demos
    print(f"Input: {inp}")
    print(f"Cleaned length (chars): {len(cleaned)}")
    print(f"Saved {len(chunks)} chunks to {chunks_dir}/ and metadata to {out_meta}\n")
    preview = min(len(chunks), args.preview)
    for i in range(preview):
        print(f"--- chunk {i} (words={metadata[i]['word_count']}) ---")
        s = (chunks[i][:1000] + ("..." if len(chunks[i]) > 1000 else ""))
        print(s)
        print()

if __name__ == "__main__":
    # CLI for configuring chunking behavior
    p = argparse.ArgumentParser(description="Chunk extracted text into overlapping word windows and write metadata.")
    p.add_argument("--input", default="docs/hr_guidelines_detailed.txt", help="path to extracted text file")
    p.add_argument("--out", default="metadata.json", help="output metadata JSON path")
    p.add_argument("--chunks_dir", default="chunks", help="directory to save chunk text files")
    p.add_argument("--chunk_size", type=int, default=120, help="chunk size in words (e.g., 120)")
    p.add_argument("--overlap", type=int, default=30, help="overlap size in words (e.g., 30)")
    p.add_argument("--preview", type=int, default=3, help="how many chunks to print as preview")
    args = p.parse_args()
    main(args)
