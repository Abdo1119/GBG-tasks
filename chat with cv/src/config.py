"""Central configuration for the CV RAG chatbot."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CVS_DIR = PROJECT_ROOT / "CVs"
VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vectors"

# Ensure directories exist
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# ── Gemini ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"          # fast + capable
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"

# ── Chunking ────────────────────────────────────────────────────────────
CHUNK_SIZE = 600          # tokens (target)
CHUNK_OVERLAP = 80        # tokens overlap between consecutive chunks

# ── Retrieval ───────────────────────────────────────────────────────────
TOP_K = 10                # chunks to retrieve per query
COLLECTION_NAME = "cv_chunks"

# ── Section headers commonly found in resumes ───────────────────────────
SECTION_HEADERS = [
    "experience", "work experience", "professional experience", "employment",
    "education", "academic background", "qualifications",
    "skills", "technical skills", "core competencies", "technologies",
    "projects", "personal projects", "key projects",
    "certifications", "certificates", "licenses",
    "summary", "profile", "objective", "about me", "about",
    "publications", "research", "awards", "honors",
    "languages", "interests", "hobbies", "volunteer",
    "references", "contact", "contact information",
    "courses", "training", "workshops",
]
