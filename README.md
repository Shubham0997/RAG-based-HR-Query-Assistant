# RAG-AI: Retrieval-Augmented Generation Pipeline

A complete RAG (Retrieval-Augmented Generation) pipeline that extracts text from PDFs, creates searchable chunks, builds a vector index, and generates AI-powered answers with citations.

## 🚀 Features

- **PDF Text Extraction**: Extract text from PDF files with comprehensive error handling
- **Smart Chunking**: Split text into overlapping chunks optimized for retrieval
- **Vector Search**: Build FAISS index for fast semantic similarity search
- **AI-Powered Q&A**: Generate accurate answers with proper citations using OpenAI's GPT models
- **Configurable Parameters**: Customize chunk sizes, overlap, and retrieval settings

## 📋 Prerequisites

- Python 3.7+
- OpenAI API key
- Required Python packages (see installation below)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd RAG-AI
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv rag-env
   # On Windows:
   rag-env\Scripts\activate
   # On macOS/Linux:
   source rag-env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install pdfplumber sentence-transformers faiss-cpu openai python-dotenv numpy
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 📖 Usage

The RAG pipeline consists of 4 sequential steps:

### Step 1: Extract Text from PDF
```bash
python extract_pdf.py hr_guidelines_detailed.pdf
```

**What happens:**
- Extracts text from all 7 pages of the PDF
- Handles pages with no extractable text (scanned images)
- Saves cleaned text to `docs/hr_guidelines_detailed.txt`
- Provides detailed progress logging and summary

**Output:**
```
🚀 Starting PDF text extraction process...
📖 Step 1: Extracting text from PDF pages...
📄 Opening PDF file: hr_guidelines_detailed.pdf
📊 Total pages in PDF: 7
🔄 Processing page 1/7... ✅ Text extracted
...
✅ PDF processed: hr_guidelines_detailed.pdf
✅ Text saved to: docs\hr_guidelines_detailed.txt
✅ Pages processed: 7
✅ Pages with text: 7
```

### Step 2: Create Text Chunks
```bash
python step2_chunking.py --input docs/hr_guidelines_detailed.txt --out metadata_hr_guidelines.json --chunks_dir chunks --chunk_size 120 --overlap 30 --preview 2
```

**What happens:**
- Cleans PDF extraction artifacts (replaces `(cid:127)` with bullets, removes number-only lines)
- Splits text into 120-word chunks with 30-word overlap
- Saves each chunk as a separate `.txt` file in the `chunks/` directory
- Creates metadata JSON mapping chunk IDs to files
- Shows preview of first 2 chunks

**Output:**
```
Input: docs\hr_guidelines_detailed.txt
Cleaned length (chars): 4297
Saved 7 chunks to chunks/ and metadata to metadata_hr_guidelines.json

--- chunk 0 (words=120) ---
Leave Policy • Employees are entitled to 20 days of annual paid leave...
--- chunk 1 (words=120) ---
Encashment of leave is possible at the end of the fiscal year...
```

### Step 3: Build Vector Index
```bash
python step3_build_index.py
```

**What happens:**
- Loads all 7 chunks and their metadata
- Encodes chunks using `all-MiniLM-L6-v2` sentence transformer model
- Creates L2-normalized embeddings (384-dimensional vectors)
- Builds FAISS index for fast cosine similarity search
- Augments metadata with vector IDs
- Runs sample query to validate the index

**Output:**
```
Loaded 7 chunks from chunks and metadata from metadata_hr_guidelines.json.
Loaded embedding model: all-MiniLM-L6-v2
Embeddings shape: (7, 384)
Wrote FAISS index (7 vectors) to faiss_index.index
Wrote metadata with vector ids to metadata_with_vecs.json

Sample retrieval for query: Can i drink alcohol during work hours?
Rank 1: idx=3, score=0.3763, doc=hr_guidelines_detailed.txt, chunk_id=3
Snippet: Alcohol and drugs are strictly prohibited during work hours...
```

### Step 4: Generate AI Answers
```bash
python generate_answer.py --index faiss_index.index --meta metadata_with_vecs.json --chunks_dir chunks --question "how many annual paid leave do we get?" --k 1
```

**What happens:**
- Encodes the question using the same embedding model
- Searches the FAISS index for the most relevant chunk(s)
- Builds a context-aware prompt with retrieved chunks
- Calls OpenAI's GPT-4o-mini model to generate a cited answer
- Returns the answer with proper citations

**Output:**
```
[main] Loading environment variables (.env if present)
[main] Reading FAISS index from: faiss_index.index
[load_meta_and_texts] Loading metadata from: metadata_with_vecs.json
[load_meta_and_texts] Metadata entries: 7
[load_meta_and_texts] Loaded text files: 7
[main] Loading embedding model: all-MiniLM-L6-v2
[retrieve_top_k] Encoding question with model: SentenceTransformer
[retrieve_top_k] top-k: 1
[retrieve_top_k] Retrieved 1 results
[build_prompt] Retrieved chunks: 1 | Context chars: 766 | Prompt chars: 1099
[call_openai_chat] model=gpt-4o-mini, max_tokens=300, temperature=0.0
[call_openai_chat] Received 59 chars of text

=== Final Answer ===

Employees are entitled to 20 days of annual paid leave [1].

=== Citations ===
[1] chunk_id=0 (doc index)
```

## 📁 Project Structure

```
RAG-AI/
├── extract_pdf.py              # PDF text extraction
├── step2_chunking.py           # Text chunking with overlap
├── step3_build_index.py        # FAISS index creation
├── generate_answer.py          # AI-powered Q&A generation
├── hr_guidelines_detailed.pdf  # Sample PDF document
├── docs/                       # Extracted text files
│   └── hr_guidelines_detailed.txt
├── chunks/                     # Individual chunk files
│   ├── 0.txt
│   ├── 1.txt
│   └── ...
├── metadata_hr_guidelines.json # Chunk metadata
├── metadata_with_vecs.json     # Metadata with vector IDs
├── faiss_index.index          # FAISS vector index
└── README.md                  # This file
```

## ⚙️ Configuration Options

### PDF Extraction (`extract_pdf.py`)
- `--output-dir`: Output directory for extracted text (default: `docs`)

### Chunking (`step2_chunking.py`)
- `--input`: Input text file path
- `--out`: Output metadata JSON path
- `--chunks_dir`: Directory for chunk files
- `--chunk_size`: Words per chunk (default: 120)
- `--overlap`: Overlap between chunks (default: 30)
- `--preview`: Number of chunks to preview (default: 3)

### Index Building (`step3_build_index.py`)
- `--chunks_dir`: Directory containing chunk files
- `--meta`: Input metadata JSON
- `--out_index`: Output FAISS index path
- `--out_meta`: Output metadata with vector IDs
- `--model`: Sentence transformer model (default: `all-MiniLM-L6-v2`)
- `--batch_size`: Embedding batch size (default: 16)
- `--sample_query`: Test query for validation

### Answer Generation (`generate_answer.py`)
- `--index`: FAISS index file path
- `--meta`: Metadata with vector IDs
- `--chunks_dir`: Directory containing chunk files
- `--question`: Question to answer
- `--k`: Number of chunks to retrieve (default: 3)
- `--emb_model`: Embedding model for queries
- `--llm_model`: OpenAI model (default: `gpt-4o-mini`)
- `--max_tokens`: Maximum answer length (default: 300)
- `--temperature`: LLM temperature (default: 0.0)

## 🔧 Advanced Usage

### Custom Questions
You can ask any question about your document:
```bash
python generate_answer.py --index faiss_index.index --meta metadata_with_vecs.json --chunks_dir chunks --question "What is the maternity leave policy?" --k 3
```

### Different Models
Use different embedding models for better performance:
```bash
python step3_build_index.py --model all-mpnet-base-v2
python generate_answer.py --emb_model all-mpnet-base-v2 --question "your question here"
```

### Batch Processing
Process multiple PDFs by running the pipeline for each:
```bash
for pdf in *.pdf; do
    python extract_pdf.py "$pdf"
    python step2_chunking.py --input "docs/${pdf%.pdf}.txt" --out "metadata_${pdf%.pdf}.json"
    python step3_build_index.py --meta "metadata_${pdf%.pdf}.json" --out_index "${pdf%.pdf}_index.index"
done
```

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**:
   ```
   RuntimeError: Set OPENAI_API_KEY environment variable before running.
   ```
   Solution: Create a `.env` file with your OpenAI API key.

2. **Missing Dependencies**:
   ```
   ModuleNotFoundError: No module named 'sentence_transformers'
   ```
   Solution: Install all required packages using pip.

3. **PDF Extraction Issues**:
   - For scanned PDFs, consider using OCR tools first
   - Ensure PDF files are not password-protected

4. **Memory Issues with Large Documents**:
   - Reduce `chunk_size` and `batch_size` parameters
   - Use smaller embedding models

## 📊 Performance Notes

- **Chunk Size**: 120 words provides good balance between context and precision
- **Overlap**: 30 words (25%) helps preserve context across chunk boundaries
- **Embedding Model**: `all-MiniLM-L6-v2` is fast and efficient for most use cases
- **Retrieval**: FAISS provides sub-second search even with thousands of chunks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [FAISS](https://faiss.ai/) for efficient similarity search
- [OpenAI](https://openai.com/) for language models
- [pdfplumber](https://github.com/jsvine/pdfplumber) for PDF text extraction
