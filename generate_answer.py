#!/usr/bin/env python3
"""
generate_answer.py

Purpose:
- Retrieve top-k relevant chunks from a local FAISS index.
- Build a strict-context prompt from retrieved chunks.
- Call an LLM to generate a concise, cited answer.

Usage (example):
    python generate_answer.py \
        --question "Can I carry my unused leaves to next year?" \
        --index faiss_index.index --meta metadata_with_vecs.json \
        --chunks_dir chunks --k 3

Notes:
- Requires OPENAI_API_KEY in environment (loadable via .env as well)
- Libraries: sentence-transformers, faiss-cpu (or faiss), openai, python-dotenv
"""

import argparse, os, json
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

def load_meta_and_texts(meta_file: Path, chunks_dir: Path):
    """
    Load chunk metadata and corresponding text content.

    - Uses the "text_file" field when available; otherwise falls back to
      chunks_dir/chunk_id.txt.
    - If a chunk text is missing, inserts an empty string but continues.
    """
    print(f"[load_meta_and_texts] Loading metadata from: {meta_file}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    print(f"[load_meta_and_texts] Metadata entries: {len(meta)}")
    texts = []
    for m in meta:
        txt_path = Path(m.get("text_file", "")) or (chunks_dir / f"{m['chunk_id']}.txt")
        if not txt_path.exists():
            txt_path = chunks_dir / f"{m['chunk_id']}.txt"
        if not txt_path.exists():
            # Warn but proceed so that missing chunks don't crash the run
            print(f"[load_meta_and_texts] WARNING: Missing text file for chunk {m.get('chunk_id')}: {txt_path}")
            texts.append("")
        else:
            texts.append(txt_path.read_text(encoding="utf-8"))
    print(f"[load_meta_and_texts] Loaded text files: {len(texts)}")
    return meta, texts

def retrieve_top_k(question, model, index, texts, k=3):
    """
    Encode the question, search the FAISS index, and return top-k results.

    Returns a list of dicts: {"score": float, "chunk_id": int, "text": str}
    """
    print(f"[retrieve_top_k] Encoding question with model: {getattr(model, 'name_or_path', type(model).__name__)}")
    print(f"[retrieve_top_k] top-k: {k}")
    q_emb = model.encode([question], convert_to_numpy=True).astype('float32')
    import faiss
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        results.append({"score": float(score), "chunk_id": int(idx), "text": texts[idx]})
    print(f"[retrieve_top_k] Retrieved {len(results)} results")
    return results

def build_prompt(question, retrieved):
    """
    Build a strict-context prompt. Each chunk is tagged with a citation [i].
    """
    context = ""
    for i, r in enumerate(retrieved, start=1):
        # include a short citation tag with each chunk
        context += f"[{i}] {r['text']}\n\n"
    print(f"[build_prompt] Context: {context}")
    prompt = f"""You are an assistant. Use ONLY the CONTEXT sections below to answer the QUESTION.
If the answer cannot be found in the context, reply exactly: "I don't know based on the provided documents."

CONTEXT:
{context}

QUESTION:
{question}

Answer concisely and include citation(s) like [1] referencing the context above.
"""
    print(f"[build_prompt] Retrieved chunks: {len(retrieved)} | Context chars: {len(context)} | Prompt chars: {len(prompt)}")
    return prompt

def call_openai_chat(prompt, model="gpt-4o-mini", max_tokens=300, temperature=0.0):
    """
    Call OpenAI Chat Completions API with a grounded prompt.
    """
    # Load environment variables (including OPENAI_API_KEY) from .env if present
    load_dotenv()
    from openai import OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY environment variable before running.")
    client = OpenAI()
    print(f"[call_openai_chat] model={model}, max_tokens={max_tokens}, temperature={temperature}")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"You are a helpful assistant that strictly uses the provided CONTEXT to answer."},
                {"role":"user","content":prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        content = resp.choices[0].message.content
        print(f"[call_openai_chat] Received {len(content)} chars of text")
        return content
    except Exception as e:
        raise RuntimeError(f"OpenAI chat completion failed: {e}")

def main(args):
    # Ensure .env is loaded early (for any other env-based configs)
    print("[main] Loading environment variables (.env if present)")
    load_dotenv()
    from sentence_transformers import SentenceTransformer
    import faiss

    # 1) Load index and metadata
    meta_file = Path(args.meta)
    chunks_dir = Path(args.chunks_dir)
    print(f"[main] Reading FAISS index from: {args.index}")
    index = faiss.read_index(str(args.index))
    meta, texts = load_meta_and_texts(meta_file, chunks_dir)

    # 2) Load embedding model for query encoding
    print(f"[main] Loading embedding model: {args.emb_model}")
    model = SentenceTransformer(args.emb_model)

    # 3) Retrieve top-k chunks
    retrieved = retrieve_top_k(args.question, model, index, texts, k=args.k)
    if not retrieved:
        print("No results found.")
        return

    # 4) Build prompt from retrieved chunks
    prompt = build_prompt(args.question, retrieved)

    # 5) Call LLM
    answer = call_openai_chat(prompt, model=args.llm_model, max_tokens=args.max_tokens, temperature=args.temperature)

    # 6) Print answer and citations
    print("\n=== Final Answer ===\n")
    print(answer)
    print("\n=== Citations ===")
    for i, r in enumerate(retrieved, start=1):
        print(f"[{i}] chunk_id={r['chunk_id']} (doc index)")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Retrieve top-k chunks and generate a cited answer via LLM.")
    p.add_argument("--index", required=True, help="path to FAISS index file")
    p.add_argument("--meta", default="metadata_with_vecs.json", help="metadata JSON with vector ids")
    p.add_argument("--chunks_dir", default="chunks", help="directory containing chunk text files")
    p.add_argument("--question", required=True, help="user question to answer")
    p.add_argument("--k", type=int, default=3, help="number of chunks to retrieve")
    p.add_argument("--emb_model", default="all-MiniLM-L6-v2", help="sentence-transformers model for query encoding")
    p.add_argument("--llm_model", default="gpt-4o-mini", help="OpenAI chat model to use")
    p.add_argument("--max_tokens", type=int, default=300, help="max tokens for the answer")
    p.add_argument("--temperature", type=float, default=0.0, help="sampling temperature for the LLM")
    args = p.parse_args()
    main(args)
