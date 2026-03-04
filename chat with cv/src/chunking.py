"""
Section-Aware Chunking with Metadata Enrichment
─────────────────────────────────────────────────
Strategy:
  1. Split each resume into logical *sections* (Experience, Education, …)
     using regex that detects common CV section headers.
  2. Within each section, create overlapping chunks of ~CHUNK_SIZE tokens
     so that no semantic boundary is lost.
  3. Attach rich metadata to every chunk for downstream filtering.

Why section-aware?
  - Preserves context boundaries (an "Experience" chunk won't bleed into
    "Education").
  - Enables metadata filtering (retrieve only "Skills" chunks).
  - Produces more coherent embeddings → better retrieval.
"""

import re
import tiktoken
from src.config import SECTION_HEADERS, CHUNK_SIZE, CHUNK_OVERLAP

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _build_section_pattern() -> re.Pattern:
    """Build a regex that matches section headers in resumes."""
    escaped = [re.escape(h) for h in SECTION_HEADERS]
    # Match line-start header with optional colon / dash / pipe
    pattern = r"(?m)^[ \t]*(?:" + "|".join(escaped) + r")[ \t]*[:\-–|]?[ \t]*$"
    return re.compile(pattern, re.IGNORECASE)


_SECTION_RE = _build_section_pattern()


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Return list of (section_name, section_text) pairs."""
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return [("general", text)]

    sections: list[tuple[str, str]] = []

    # Text before the first section header
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("summary", preamble))

    for i, m in enumerate(matches):
        header = m.group().strip().rstrip(":-–| \t").strip().lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((_normalise_section(header), body))

    return sections if sections else [("general", text)]


def _normalise_section(header: str) -> str:
    """Map variations to canonical section names."""
    h = header.lower().strip()
    mapping = {
        "work experience": "experience",
        "professional experience": "experience",
        "employment": "experience",
        "academic background": "education",
        "qualifications": "education",
        "technical skills": "skills",
        "core competencies": "skills",
        "technologies": "skills",
        "personal projects": "projects",
        "key projects": "projects",
        "certificates": "certifications",
        "licenses": "certifications",
        "profile": "summary",
        "objective": "summary",
        "about me": "summary",
        "about": "summary",
        "contact information": "contact",
    }
    return mapping.get(h, h)


def _extract_skill_tags(text: str) -> list[str]:
    """Extract likely skill keywords from chunk text using word-boundary matching."""
    import re as _re

    known_skills = [
        "python", "java", "javascript", "typescript", "c\\+\\+", "c#", "golang", "rust",
        "sql", "nosql", "mongodb", "postgresql", "mysql", "redis",
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "react", "angular", "vue", "node\\.js", "django", "flask", "fastapi",
        "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
        "machine learning", "deep learning", "nlp", "computer vision",
        "data science", "data engineering", "mlops", "devops", "ci/cd",
        "git", "linux", "agile", "scrum", "rest api", "graphql",
        "html", "css", "sass", "tailwind",
        "spark", "hadoop", "airflow", "kafka",
        "power bi", "tableau", "excel",
        "figma", "photoshop",
        "natural language processing", "reinforcement learning",
        "generative ai", "llm", "rag", "langchain", "openai",
        "selenium", "cypress", "jest", "pytest",
    ]
    lower = text.lower()
    found = []
    for skill in known_skills:
        # Use word boundaries to avoid false positives
        if _re.search(r"(?<![a-z])" + skill + r"(?![a-z])", lower):
            # Store the clean display name (un-escape regex)
            found.append(skill.replace("\\", ""))
    return sorted(set(found))


def _extract_years_of_experience(text: str) -> str | None:
    """Try to extract years-of-experience mentions."""
    m = re.search(r"(\d{1,2})\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience", text, re.IGNORECASE)
    return m.group(1) if m else None


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping token-level chunks."""
    tokens = _enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(_enc.decode(chunk_tokens))
        start += chunk_size - overlap
    return chunks


def chunk_documents(docs: list[dict]) -> list[dict]:
    """
    Take raw documents and produce enriched chunks.

    Each chunk dict:
        id, text, candidate_name, source_file, section_type,
        skill_tags, years_of_experience
    """
    all_chunks: list[dict] = []
    chunk_id = 0

    for doc in docs:
        sections = _split_into_sections(doc["text"])

        for section_name, section_text in sections:
            text_chunks = _chunk_text(section_text)

            for chunk_text in text_chunks:
                skills = _extract_skill_tags(chunk_text)
                yoe = _extract_years_of_experience(chunk_text)

                all_chunks.append({
                    "id": f"chunk_{chunk_id:04d}",
                    "text": chunk_text,
                    "candidate_name": doc["candidate_name"],
                    "source_file": doc["source_file"],
                    "section_type": section_name,
                    "skill_tags": ", ".join(skills) if skills else "",
                    "years_of_experience": yoe or "",
                })
                chunk_id += 1

    return all_chunks
