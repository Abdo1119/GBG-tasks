"""
نظام استرجاع معزز بالتوليد (RAG) للنصوص العربية
BAAI/bge-m3 + ChromaDB + Gemini + Streamlit

Usage:
    streamlit run rag_arabic.py
"""

import sys
import re
import os
import hashlib

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ─── إعدادات عامة ────────────────────────────────────────────────────────────

TEXT_FILE = "arabic.txt"
MODEL_NAME = "BAAI/bge-m3"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "arabic_rag"
CHUNK_MIN_CHARS = 400
CHUNK_MAX_CHARS = 800
OVERLAP_SENTENCES = 1

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyAtV7r2_u0VzLvjC8KHq_wkTD_ohlDZ0gg")
GEMINI_MODEL = "gemini-2.0-flash"


# ─── تحميل وتنظيف النص ───────────────────────────────────────────────────────

def load_text(path: str) -> str:
    """تحميل ملف نصي بترميز UTF-8."""
    if not os.path.isfile(path):
        st.error(f"الملف '{path}' غير موجود!")
        st.stop()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text: str) -> str:
    """تنظيف النص."""
    text = text.strip()
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def file_hash(path: str) -> str:
    """حساب hash للملف."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# ─── تقسيم النص إلى مقاطع ────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.۔。؟?!؛\n])\s*", text)
    return [s.strip() for s in parts if s.strip()]


def split_into_chunks(
    text: str,
    min_chars: int = CHUNK_MIN_CHARS,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap: int = OVERLAP_SENTENCES,
) -> list[str]:
    """تقسيم النص إلى مقاطع متداخلة."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    result_chunks: list[str] = []
    i = 0

    while i < len(sentences):
        current_chunk_sents: list[str] = []
        current_len = 0

        while i < len(sentences):
            sent = sentences[i]
            new_len = current_len + len(sent) + (1 if current_chunk_sents else 0)

            if new_len > max_chars and current_len >= min_chars:
                break

            current_chunk_sents.append(sent)
            current_len = new_len
            i += 1

            if current_len >= min_chars and i < len(sentences):
                next_len = current_len + len(sentences[i]) + 1
                if next_len > max_chars:
                    break

        if current_chunk_sents:
            result_chunks.append(" ".join(current_chunk_sents))

        if i < len(sentences) and overlap > 0:
            i -= min(overlap, len(current_chunk_sents) - 1)

    return result_chunks


# ─── التمثيل المتجهي + ChromaDB ──────────────────────────────────────────────

def encode(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """تحويل نصوص إلى متجهات."""
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return emb.tolist()


def init_chroma() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=CHROMA_DIR)


def build_index(
    client: chromadb.ClientAPI,
    model: SentenceTransformer,
    text_chunks: list[str],
    text_hash: str,
) -> chromadb.Collection:
    """بناء أو استرجاع فهرس ChromaDB."""
    existing = [c.name for c in client.list_collections()]

    if COLLECTION_NAME in existing:
        coll = client.get_collection(COLLECTION_NAME)
        stored_meta = coll.metadata or {}
        if stored_meta.get("file_hash") == text_hash:
            return coll
        else:
            client.delete_collection(COLLECTION_NAME)

    coll = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"file_hash": text_hash},
    )

    embeddings = encode(model, text_chunks)
    ids = [f"chunk_{i}" for i in range(len(text_chunks))]
    metadatas = [{"chunk_id": i, "char_len": len(c)} for i, c in enumerate(text_chunks)]

    coll.add(ids=ids, embeddings=embeddings, documents=text_chunks, metadatas=metadatas)
    return coll


# ─── البحث ────────────────────────────────────────────────────────────────────

def search(
    model: SentenceTransformer,
    collection: chromadb.Collection,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """البحث في ChromaDB."""
    query_emb = model.encode([query], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=query_emb,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        score = 1.0 - distance

        output.append({
            "chunk_id": results["metadatas"][0][i]["chunk_id"],
            "score": score,
            "text": results["documents"][0][i],
        })
    return output


# ─── Gemini للإجابة الذكية ────────────────────────────────────────────────────

def ask_gemini(query: str, retrieved_chunks: list[dict]) -> str:
    """إرسال السؤال والمقاطع المسترجعة إلى Gemini للحصول على إجابة."""
    context = ""
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context += f"[مقطع {i}]: {chunk['text']}\n\n"

    prompt = f"""أنت مساعد ذكي متخصص في الإجابة عن الأسئلة بالعربية.
استخدم المقاطع التالية فقط للإجابة. لا تضف معلومات من خارج النص.
إذا لم تجد إجابة في المقاطع، قل "لم أجد إجابة في النص المتاح".

المقاطع:
{context}

السؤال: {query}

أجب بالعربية إجابة واضحة ومختصرة، مع الإشارة لأرقام المقاطع المستخدمة."""

    genai.configure(api_key=GEMINI_API_KEY)
    gemini = genai.GenerativeModel(
        GEMINI_MODEL,
        generation_config=genai.GenerationConfig(
            max_output_tokens=500,
            temperature=0.3,
        ),
    )
    response = gemini.generate_content(prompt)
    return response.text


# ─── تحميل الموارد (cached) ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="جارٍ تحميل النموذج...")
def load_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource(show_spinner="جارٍ بناء الفهرس...")
def load_index():
    model = load_model()
    raw = load_text(TEXT_FILE)
    text = clean_text(raw)
    text_h = file_hash(TEXT_FILE)
    chunks = split_into_chunks(text)
    client = init_chroma()
    coll = build_index(client, model, chunks, text_h)
    return model, coll, chunks


# ─── واجهة Streamlit ─────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="RAG — القلعة البيضاء",
        page_icon="🏰",
        layout="centered",
    )

    # RTL support
    st.markdown("""
    <style>
        .stApp { direction: rtl; }
        .stTextInput input, .stTextArea textarea { direction: rtl; text-align: right; }
        .stMarkdown { direction: rtl; text-align: right; }
        .result-box {
            border-right: 4px solid;
            padding: 12px 16px;
            margin: 8px 0;
            border-radius: 8px;
            line-height: 1.8;
        }
        .score-high { border-color: #22c55e; background: rgba(34,197,94,0.08); }
        .score-mid  { border-color: #eab308; background: rgba(234,179,8,0.08); }
        .score-low  { border-color: #ef4444; background: rgba(239,68,68,0.08); }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align:center;'>🏰 نظام الاسترجاع المعزز — القلعة البيضاء</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:gray;'>BAAI/bge-m3 + ChromaDB + Gemini</p>", unsafe_allow_html=True)

    # تحميل النموذج والفهرس
    model, collection, chunks = load_index()

    st.markdown("---")

    # خانة السؤال
    query = st.text_input("اكتب سؤالك عن القلعة البيضاء:", placeholder="مثال: ما هو النظام الهيدروليكي في القلعة؟")

    col1, col2 = st.columns([3, 1])
    with col2:
        top_k = st.slider("عدد النتائج", 1, 7, 3)

    if query:
        # البحث
        with st.spinner("جارٍ البحث..."):
            results = search(model, collection, query, top_k=top_k)

        # إجابة Gemini
        with st.spinner("جارٍ توليد الإجابة من Gemini..."):
            try:
                answer = ask_gemini(query, results)
                st.markdown("### 💡 الإجابة")
                st.markdown(f"<div style='background:#1a1a2e; padding:16px; border-radius:10px; line-height:2;'>{answer}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"خطأ في Gemini: {e}")

        # عرض المقاطع المسترجعة
        st.markdown("### 📄 المقاطع المسترجعة")

        for i, r in enumerate(results, start=1):
            score = r["score"]
            if score >= 0.5:
                css_class = "score-high"
            elif score >= 0.3:
                css_class = "score-mid"
            else:
                css_class = "score-low"

            st.markdown(
                f"""<div class="result-box {css_class}">
                <strong>المقطع {i}</strong> — رقم: {r['chunk_id']} | درجة التشابه: {score:.4f}<br><br>
                {r['text']}
                </div>""",
                unsafe_allow_html=True,
            )

    # معلومات النظام
    with st.sidebar:
        st.markdown("### معلومات النظام")
        st.markdown(f"- **عدد المقاطع:** {len(chunks)}")
        st.markdown(f"- **النموذج:** `{MODEL_NAME}`")
        st.markdown(f"- **LLM:** `{GEMINI_MODEL}`")
        st.markdown(f"- **قاعدة البيانات:** ChromaDB")
        st.markdown("---")
        st.markdown("**أمثلة أسئلة:**")
        st.markdown("- ما هو النظام الهيدروليكي في القلعة؟")
        st.markdown("- ماذا حدث في العهد العثماني؟")
        st.markdown("- ما هي المكتبة السرية؟")
        st.markdown("- متى تم ترميم القلعة؟")


if __name__ == "__main__":
    main()
