"""
LLM Reasoning Layer – Prompt Construction & Answer Generation
──────────────────────────────────────────────────────────────
Uses structured prompts with few-shot examples to guide Gemini
in answering questions accurately based on retrieved resume chunks.
"""

from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY, GEMINI_MODEL

_client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """You are an expert HR assistant specializing in resume analysis.
You answer questions about candidates based ONLY on the resume excerpts provided below.

RULES:
1. Base your answers strictly on the provided resume excerpts.
2. If information is not in the excerpts, say "I don't have enough information in the provided resumes to answer that."
3. When comparing candidates, use structured tables or bullet points.
4. Always mention which candidate you're referring to by name.
5. Be specific — cite concrete skills, companies, years, or degrees from the resumes.
6. If asked to summarize, be concise but comprehensive.
7. NEVER reference internal excerpt numbers like "Excerpt 1", "Excerpt 3", etc. in your answer. The user cannot see these. Instead, just mention the candidate name and the actual details directly.

FEW-SHOT EXAMPLES:

Q: Which candidate has the most experience in machine learning?
A: Based on the resumes:
• **Ahmed Hassan** — 4+ years of ML experience at DataCorp, worked on NLP models and recommendation systems using PyTorch and scikit-learn.
• **Sara Ali** — 2 years of ML experience, primarily in computer vision using TensorFlow.
→ **Ahmed Hassan** has the most ML experience (4+ years vs 2 years).

Q: Compare candidates' education backgrounds.
A: | Candidate | Degree | University | Year |
   |-----------|--------|-----------|------|
   | Ahmed Hassan | MSc Computer Science | Cairo University | 2020 |
   | Sara Ali | BSc Data Science | AUC | 2021 |

Q: Extract years of experience for each candidate.
A: • **Ahmed Hassan**: 5 years (2019–2024) — ML Engineer at DataCorp, then Senior ML Engineer at TechFlow
   • **Sara Ali**: 3 years (2021–2024) — Data Scientist at Analytics Inc.
"""


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a structured context block."""
    if not chunks:
        return "No relevant resume excerpts found."

    context_parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        header = (
            f"[Excerpt {i}] Candidate: {chunk.get('candidate_name', 'Unknown')} "
            f"| Section: {chunk.get('section_type', 'general')}"
        )
        skills = chunk.get("skill_tags", "")
        if skills:
            header += f" | Skills: {skills}"
        context_parts.append(f"{header}\n{chunk['text']}")

    return "\n\n---\n\n".join(context_parts)


def generate_answer(query: str, chunks: list[dict], chat_history: list[dict] | None = None) -> str:
    """
    Generate an answer using Gemini with retrieved context.

    Args:
        query: the user's question
        chunks: retrieved resume chunks
        chat_history: list of {"role": "user"|"assistant", "content": str}
    """
    context = build_context(chunks)

    # Build chat messages ensuring strict user/model alternation
    contents: list[types.Content] = []

    if chat_history:
        prev_role = None
        for msg in chat_history[-6:]:
            role = "user" if msg["role"] == "user" else "model"
            if role == prev_role:
                continue
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
            prev_role = role

        # Ensure last history message is from model (so our next is user)
        if contents and contents[-1].role == "user":
            contents.pop()

    # Build the current user message with context
    user_message = f"""RESUME EXCERPTS:
{context}

QUESTION: {query}

Answer the question based on the resume excerpts above. Be specific and cite details from the resumes."""

    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))

    response = _client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        ),
    )
    return response.text or "I couldn't generate a response. Please try rephrasing your question."
