"""
CV RAG Chatbot – Streamlit Interface
─────────────────────────────────────
A conversational chatbot for querying and comparing up to 5 resumes
using Retrieval-Augmented Generation.
"""

import streamlit as st

from src.config import CVS_DIR
from src.pipeline import build_index
from src.retrieval import advanced_retrieve
from src.llm import generate_answer
from src.vectorstore import is_indexed

# ── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CV Chatbot – RAG Resume Analyzer",
    page_icon="📄",
    layout="wide",
)

# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stChatMessage { max-width: 900px; }
    .block-container { max-width: 1000px; margin: auto; }
    div[data-testid="stSidebarContent"] { padding-top: 1rem; }
    .chunk-debug {
        font-size: 0.8em; color: #888;
        border-left: 3px solid #ddd;
        padding-left: 10px; margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ──────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed" not in st.session_state:
    st.session_state.indexed = is_indexed()
if "index_info" not in st.session_state:
    st.session_state.index_info = None
if "show_sources" not in st.session_state:
    st.session_state.show_sources = False
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


def _process_question(question: str):
    """Run the RAG pipeline for a question and store the result."""
    chunks = advanced_retrieve(question)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    answer = generate_answer(question, chunks, chat_history=history)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": chunks[:5] if chunks else [],
    })


# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 CV RAG Chatbot")
    st.markdown("---")

    # Index Management
    st.subheader("Index Management")

    cv_files = list(CVS_DIR.glob("*.pdf")) + list(CVS_DIR.glob("*.docx"))
    st.markdown(f"**Found {len(cv_files)} CVs** in `CVs/` folder:")
    for f in cv_files:
        st.markdown(f"- {f.name}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔨 Build Index", use_container_width=True):
            with st.spinner("Ingesting, chunking & embedding..."):
                try:
                    info = build_index(CVS_DIR, force=True)
                    st.session_state.indexed = True
                    st.session_state.index_info = info
                    st.success(
                        f"Indexed {info['documents']} CVs → {info['chunks']} chunks"
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if st.session_state.index_info and st.session_state.index_info.get("candidates"):
        st.markdown("---")
        st.subheader("Candidates")
        for name in st.session_state.index_info["candidates"]:
            st.markdown(f"- **{name}**")

    st.markdown("---")
    st.session_state.show_sources = st.checkbox(
        "Show retrieved sources", value=False
    )

    st.markdown("---")
    st.markdown("### Example Questions")
    examples = [
        "Which candidate has the most experience in machine learning?",
        "Which candidates have Python + NLP experience?",
        "Compare all candidates' education backgrounds",
        "Who has worked with AWS or cloud technologies?",
        "Summarize each candidate in 3 bullet points",
        "Extract years of experience for each candidate",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{hash(ex)}", use_container_width=True):
            st.session_state.pending_question = ex
            st.rerun()


# ── Main Chat Area ──────────────────────────────────────────────────────
st.header("💬 Chat with your CVs")

if not st.session_state.indexed:
    st.info(
        "👈 Click **Build Index** in the sidebar to index your CVs before chatting."
    )

# Handle pending question from example buttons
if st.session_state.pending_question and st.session_state.indexed:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
    with st.spinner("Thinking..."):
        try:
            _process_question(question)
        except Exception as e:
            st.error(f"Error generating answer: {e}")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if (
            msg["role"] == "assistant"
            and st.session_state.show_sources
            and "sources" in msg
        ):
            with st.expander("📎 Retrieved Sources"):
                for src in msg["sources"]:
                    st.markdown(
                        f"<div class='chunk-debug'>"
                        f"<b>{src['candidate_name']}</b> | "
                        f"{src['section_type']} "
                        f"| score: {src.get('score', 0):.3f}<br>"
                        f"{src['text'][:200]}..."
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# Chat input
if prompt := st.chat_input("Ask about the candidates..."):
    if not st.session_state.indexed:
        st.warning("Please build the index first using the sidebar button.")
    else:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    chunks = advanced_retrieve(prompt)
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ]
                    answer = generate_answer(prompt, chunks, chat_history=history)
                    st.markdown(answer)

                    if st.session_state.show_sources and chunks:
                        with st.expander("📎 Retrieved Sources"):
                            for src in chunks[:5]:
                                st.markdown(
                                    f"<div class='chunk-debug'>"
                                    f"<b>{src['candidate_name']}</b> | "
                                    f"{src['section_type']} "
                                    f"| score: {src.get('score', 0):.3f}<br>"
                                    f"{src['text'][:200]}..."
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                except Exception as e:
                    answer = f"Sorry, an error occurred: {e}"
                    st.error(answer)
                    chunks = []

        # Save messages to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": chunks[:5] if chunks else [],
        })
