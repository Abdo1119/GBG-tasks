"""
Pipeline Orchestrator
─────────────────────
Ties together ingestion → chunking → indexing in one call.
"""

from pathlib import Path
from src.ingestion import load_documents
from src.chunking import chunk_documents
from src.vectorstore import index_chunks, is_indexed


def build_index(cv_dir: Path, force: bool = False) -> dict:
    """
    Run the full ingestion pipeline.

    Returns:
        {"documents": int, "chunks": int, "candidates": list[str]}
    """
    if is_indexed() and not force:
        return {"status": "already_indexed"}

    # Step 1: Load documents
    docs = load_documents(cv_dir)
    if not docs:
        raise FileNotFoundError(f"No PDF/DOCX files found in {cv_dir}")

    # Step 2: Section-aware chunking with metadata
    chunks = chunk_documents(docs)

    # Step 3: Embed and index
    n_indexed = index_chunks(chunks)

    candidates = list({d["candidate_name"] for d in docs})
    return {
        "status": "indexed",
        "documents": len(docs),
        "chunks": n_indexed,
        "candidates": sorted(candidates),
    }
