"""
Document Ingestion Pipeline
────────────────────────────
Extracts raw text from PDF and DOCX resumes, normalises whitespace,
and uses LLM-based name extraction for accurate candidate identification.
"""

from pathlib import Path
import re
import pdfplumber
from docx import Document as DocxDocument
from google import genai
from src.config import GEMINI_API_KEY, GEMINI_MODEL

_client = genai.Client(api_key=GEMINI_API_KEY)


def extract_text_from_pdf(path: Path) -> str:
    """Extract text from a PDF using pdfplumber (handles multi-column layouts)."""
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


def extract_text_from_docx(path: Path) -> str:
    """Extract text from a DOCX file."""
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def normalise_text(text: str) -> str:
    """Clean whitespace, control chars, and normalise line endings."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)           # collapse horizontal ws
    text = re.sub(r"\n{3,}", "\n\n", text)           # max 2 newlines
    return text.strip()


def _extract_candidate_name(text: str, filename: str) -> str:
    """
    Use the LLM to extract the candidate's full name from the resume text.
    Falls back to heuristic if the LLM call fails.
    """
    # Send only the first ~500 chars — the name is always near the top
    head = text[:500]

    prompt = (
        "Extract ONLY the candidate's full name from this resume excerpt. "
        "Return nothing but the name — no titles, no degrees, no job roles, "
        "no extra text, no quotes.\n\n"
        f"{head}"
    )

    try:
        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        name = (response.text or "").strip().strip('"').strip("'")
        # Validate: should be 1-5 words, no weird chars
        if name and 1 <= len(name.split()) <= 5 and not any(c in name for c in "{}[]()@#$"):
            return name
    except Exception:
        pass

    # Fallback: simple heuristic
    return _heuristic_name(text, filename)


def _heuristic_name(text: str, filename: str) -> str:
    """Fallback heuristic: first short non-label line, else filename."""
    for line in text.split("\n"):
        line = line.strip()
        if line and len(line) < 60 and not line.lower().startswith(
            ("http", "email", "phone", "address")
        ):
            name = re.sub(r"\s*[-–|].*$", "", line).strip()
            if name and len(name.split()) <= 5:
                return name
    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


def load_documents(cv_dir: Path) -> list[dict]:
    """
    Load all PDF/DOCX files from *cv_dir*.

    Returns a list of dicts:
        { "text": str, "candidate_name": str, "source_file": str }
    """
    docs = []
    for path in sorted(cv_dir.iterdir()):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            raw = extract_text_from_pdf(path)
        elif suffix in (".docx", ".doc"):
            raw = extract_text_from_docx(path)
        else:
            continue

        text = normalise_text(raw)
        if not text:
            continue

        candidate = _extract_candidate_name(text, path.name)
        docs.append({
            "text": text,
            "candidate_name": candidate,
            "source_file": path.name,
        })
    return docs
