"""
Advanced Retrieval Pipeline
────────────────────────────
Techniques applied:
  1. **Multi-query generation** – the LLM rewrites the user query into 2-3
     diverse sub-queries to capture different phrasings and angles.
  2. **Metadata-aware filtering** – if the query targets a specific section
     (e.g. "skills"), we add a ChromaDB `where` filter.
  3. **Reciprocal Rank Fusion (RRF)** – merges results from multiple queries
     into a single ranked list, reducing bias from any single query phrasing.
"""

import re
from google import genai
from src.config import GEMINI_API_KEY, GEMINI_MODEL, TOP_K
from src.vectorstore import retrieve

_client = genai.Client(api_key=GEMINI_API_KEY)


def _generate_sub_queries(user_query: str) -> list[str]:
    """Use the LLM to generate 2-3 alternative queries."""
    prompt = f"""You are a helpful assistant that generates search queries for a resume database.
Given the user question below, generate 3 different search queries that would help find
relevant information in candidate resumes. Each query should approach the question from
a different angle or use different keywords.

User question: {user_query}

Return ONLY the queries, one per line, numbered 1-3. No explanations."""

    try:
        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        text = (response.text or "").strip()
    except Exception:
        return [user_query]

    queries = []
    for line in text.split("\n"):
        line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
        if line:
            queries.append(line)

    return queries[:3] if queries else [user_query]


def _detect_section_filter(query: str) -> dict | None:
    """
    Detect if the query targets a specific resume section.
    Only filters for narrow, section-specific queries.
    Broad comparison queries should search all sections.
    """
    q = query.lower()

    # Skip filtering for broad/comparison queries
    broad_signals = [
        "compare", "all candidates", "each candidate", "summarize",
        "who has", "which candidate", "rank", "list all", "everyone",
        "most experience", "years of experience",
    ]
    if any(signal in q for signal in broad_signals):
        return None

    section_keywords = {
        "skills": ["skill", "technology", "tech stack", "proficien", "tools"],
        "education": ["education", "degree", "university", "college", "academic", "school"],
        "certifications": ["certif", "license", "credential"],
    }
    for section, keywords in section_keywords.items():
        if any(kw in q for kw in keywords):
            return {"section_type": section}
    return None


def _reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Merge multiple ranked result lists using RRF."""
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
            doc_map[doc_id] = doc

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[did] for did in sorted_ids]


def advanced_retrieve(user_query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Full retrieval pipeline:
      1. Generate sub-queries
      2. Detect section filters
      3. Retrieve for each sub-query
      4. Fuse results with RRF
      5. Return top_k results
    """
    sub_queries = _generate_sub_queries(user_query)
    section_filter = _detect_section_filter(user_query)

    all_results: list[list[dict]] = []
    for sq in sub_queries:
        hits = retrieve(sq, top_k=top_k, where=section_filter)
        all_results.append(hits)

    # Also retrieve with the original query (no rewriting)
    original_hits = retrieve(user_query, top_k=top_k, where=section_filter)
    all_results.append(original_hits)

    fused = _reciprocal_rank_fusion(all_results)
    return fused[:top_k]
