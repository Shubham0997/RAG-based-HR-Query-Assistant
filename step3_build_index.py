#!/usr/bin/env python3
# step3_build_index.py
"""
Step 3: Build a FAISS index over chunk embeddings.

This script:
- Loads chunk texts and their metadata (produced in step 2)
- Encodes each chunk into a vector using a SentenceTransformer model
- L2-normalizes embeddings and builds an Inner Product (cosine) FAISS index
- Writes the index to disk and augments metadata with vector ids
- Optionally runs a sample query to sanity-check retrieval

Why FAISS and normalization?
- With L2-normalization, inner product is equivalent to cosine similarity
  which works well for semantic search.
"""

import argparse, json
from pathlib import Path
import numpy as np

def load_chunks_and_meta(chunks_dir: Path, meta_file: Path):
    """
    Load metadata JSON and associated chunk text files.

    Falls back to resolving chunk files relative to chunks_dir if the path
    stored in metadata is not found.
    """
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    texts = []
    for m in meta:
        txt_path = Path(m["text_file"])
        if not txt_path.exists():
            txt_path = chunks_dir / f"{m['chunk_id']}.txt"
        if not txt_path.exists():
            raise FileNotFoundError(f"Chunk file not found: {txt_path}")
        texts.append(txt_path.read_text(encoding="utf-8"))
    return meta, texts

def main(args):
    # Lazy imports to keep CLI responsiveness fast
    from sentence_transformers import SentenceTransformer
    import faiss

    # Resolve paths
    chunks_dir = Path(args.chunks_dir)
    meta_file = Path(args.meta)
    out_index = Path(args.out_index)
    out_meta = Path(args.out_meta)

    # 1) Load chunk texts and metadata
    meta, texts = load_chunks_and_meta(chunks_dir, meta_file)
    print(f"Loaded {len(texts)} chunks from {chunks_dir} and metadata from {meta_file}.")

    if len(texts) == 0:
        raise RuntimeError("No chunks found to embed. Ensure step2 produced chunks and metadata.")

    # 2) Encode chunks into embeddings in batches
    model = SentenceTransformer(args.model)
    print(f"Loaded embedding model: {args.model}")
    batch_size = max(1, args.batch_size)
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=True)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings).astype('float32')
    print("Embeddings shape:", embeddings.shape)

    # 3) Normalize and build FAISS index (cosine via inner product)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(out_index))
    print(f"Wrote FAISS index ({index.ntotal} vectors) to {out_index}")

    # 4) Add vector ids to metadata and persist
    for i, m in enumerate(meta):
        m["vector_id"] = i
        # include optional fields for traceability
        m.setdefault("embedding_model", args.model)
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote metadata with vector ids to {out_meta}")

    # 5) Optional: run a quick sample retrieval to validate the index
    if args.sample_query:
        q = args.sample_query
        q_emb = model.encode([q], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, args.k)
        print("\nSample retrieval for query:", q)
        for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
            if idx < 0:
                continue
            print(f"Rank {rank}: idx={idx}, score={float(score):.4f}, doc={meta[idx]['doc']}, chunk_id={meta[idx]['chunk_id']}")
            print("Snippet:", texts[idx][:3000].replace("\n", " "), "\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build a FAISS index over chunk embeddings.")
    p.add_argument("--chunks_dir", default="chunks", help="directory containing chunk .txt files")
    p.add_argument("--meta", default="metadata_hr_guidelines.json", help="metadata JSON produced in step2")
    p.add_argument("--out_index", default="faiss_index.index", help="output path for FAISS index")
    p.add_argument("--out_meta", default="metadata_with_vecs.json", help="output path for augmented metadata")
    p.add_argument("--model", default="all-MiniLM-L6-v2", help="sentence-transformers model name")
    p.add_argument("--batch_size", type=int, default=16, help="embedding batch size")
    p.add_argument("--k", type=int, default=3, help="top-k for optional sample retrieval")
    p.add_argument("--sample_query", default="Can i drink alcohol during work hours?", help="if provided, run a sample retrieval")
    args = p.parse_args()
    main(args)
