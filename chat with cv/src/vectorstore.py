"""
Vector Store – Embedding + ChromaDB Indexing
─────────────────────────────────────────────
Uses Google's text-embedding-004 via the google.genai SDK for embeddings
and ChromaDB with HNSW (cosine) for approximate nearest-neighbour search.

Why Gemini embeddings?
  - 768-dim, fast, and free-tier friendly.
  - Same provider as the LLM → simpler infra.

Why ChromaDB?
  - Zero-config local vector DB with built-in HNSW.
  - Supports metadata filtering out of the box.
  - Persists to disk → survives restarts.
"""

import time
from google import genai
import chromadb
from chromadb.config import Settings

from src.config import (
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_MODEL,
    VECTOR_DB_DIR,
    COLLECTION_NAME,
    TOP_K,
)

# ── Configure Gemini client ────────────────────────────────────────────
_client = genai.Client(api_key=GEMINI_API_KEY)


def _embed_texts(texts: list[str], batch_size: int = 20) -> list[list[float]]:
    """Embed a list of texts using Gemini, with batching and rate-limit handling."""
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(5):
            try:
                result = _client.models.embed_content(
                    model=GEMINI_EMBEDDING_MODEL,
                    contents=batch,
                )
                for emb in result.embeddings:
                    all_embeddings.append(emb.values)
                break
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait = 2 ** attempt
                    time.sleep(wait)
                else:
                    raise
        # Small delay between batches to respect rate limits
        if i + batch_size < len(texts):
            time.sleep(0.3)
    return all_embeddings


def _embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    result = _client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=text,
    )
    return result.embeddings[0].values


def _get_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection."""
    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity via HNSW
    )
    return collection


def index_chunks(chunks: list[dict]) -> int:
    """
    Embed and index all chunks into ChromaDB.
    Returns the number of chunks indexed.
    """
    collection = _get_collection()

    # Clear existing data for a fresh index
    existing = collection.count()
    if existing > 0:
        collection.delete(ids=collection.get()["ids"])

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [
        {
            "candidate_name": c["candidate_name"],
            "source_file": c["source_file"],
            "section_type": c["section_type"],
            "skill_tags": c["skill_tags"],
            "years_of_experience": c["years_of_experience"],
        }
        for c in chunks
    ]

    embeddings = _embed_texts(texts)

    # Index in batches
    batch = 50
    for i in range(0, len(ids), batch):
        collection.add(
            ids=ids[i : i + batch],
            embeddings=embeddings[i : i + batch],
            documents=texts[i : i + batch],
            metadatas=metadatas[i : i + batch],
        )

    return len(ids)


def retrieve(query: str, top_k: int = TOP_K, where: dict | None = None) -> list[dict]:
    """
    Retrieve the most relevant chunks for a query.

    Args:
        query:  natural-language question
        top_k:  number of results
        where:  optional ChromaDB metadata filter

    Returns list of dicts with keys: text, candidate_name, section_type, score, …
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    query_embedding = _embed_query(query)

    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    hits: list[dict] = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "score": 1 - results["distances"][0][i],  # cosine distance → similarity
            **results["metadatas"][0][i],
        })
    return hits


def is_indexed() -> bool:
    """Check if the vector store already has data."""
    try:
        return _get_collection().count() > 0
    except Exception:
        return False
