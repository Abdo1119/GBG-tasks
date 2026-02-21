# RAG Overview — القلعة البيضاء 🏰

A minimal **Retrieval-Augmented Generation (RAG)** pipeline over Arabic text, built as a learning project.

## What It Does

You ask a question in Arabic about "القلعة البيضاء" (The White Castle) → the system finds the most relevant text passages → Gemini generates a smart answer based on those passages.

## Architecture

```
arabic.txt → Clean → Sentence Split → 7 Chunks (400-800 chars, 1-sentence overlap)
                                          ↓
                                   BGE-M3 Encode (embeddings)
                                          ↓
                                   ChromaDB (persistent vector store)
                                          ↓
                          User Query → BGE-M3 Encode → ChromaDB Search
                                          ↓
                                   Top-K Relevant Chunks
                                          ↓
                              Chunks + Query → Gemini 2.0 Flash
                                          ↓
                                   Arabic Answer + Citations
                                          ↓
                                   Streamlit Web UI (RTL)
```

## Tech Stack

| Component | Tool |
|-----------|------|
| Embedding Model | `BAAI/bge-m3` (multilingual, strong on Arabic) |
| Vector Database | ChromaDB (persistent, local) |
| LLM | Gemini 2.0 Flash (via Google AI API) |
| Web UI | Streamlit (RTL support) |
| Language | Python 3.10+ |

## Pipeline Steps

### 1. Text Loading & Cleaning
- Loads `arabic.txt` (UTF-8)
- Normalizes whitespace, removes extra blank lines

### 2. Chunking
- Splits text into sentences using Arabic punctuation (`.` `؟` `!` `؛`)
- Groups sentences into chunks of 400–800 characters
- 1-sentence overlap between consecutive chunks to preserve context
- Result: **7 chunks** from 21 sentences

### 3. Embedding & Indexing
- Each chunk is encoded into a 1024-dim vector using `BAAI/bge-m3`
- Vectors are normalized (`normalize_embeddings=True`) so dot product = cosine similarity
- Stored in **ChromaDB** (persistent on disk in `./chroma_db/`)
- File hash tracking: re-encodes only when `arabic.txt` changes

### 4. Retrieval
- User query is encoded with the same model
- ChromaDB finds the closest chunks by vector similarity
- Returns top-k results with similarity scores

### 5. Generation
- Retrieved chunks + user query are sent to **Gemini 2.0 Flash**
- Prompt instructs Gemini to answer in Arabic using only the provided chunks
- `max_output_tokens=500`, `temperature=0.3` for concise, focused answers

### 6. Web Interface
- Streamlit app with full RTL (right-to-left) support
- Color-coded results: 🟢 high similarity | 🟡 medium | 🔴 low
- Sidebar with system info and example questions

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run rag_arabic.py
```

> First run downloads the BGE-M3 model (~2GB). After that, embeddings are cached in ChromaDB.

## Files

```
RAG-overview/
├── rag_arabic.py      # Main RAG pipeline + Streamlit UI
├── arabic.txt          # Source text (Arabic, about القلعة البيضاء)
├── english.txt         # English version of the text
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Example Questions

- ما هو النظام الهيدروليكي في القلعة؟
- ماذا حدث في العهد العثماني؟
- ما هي المكتبة السرية؟
- متى تم ترميم القلعة؟
