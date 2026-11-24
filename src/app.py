import os
import re
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Ensure module import resolution from any working directory
SRC_DIR = str(Path(__file__).resolve().parent)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

load_dotenv()

from rag_engine import ISROFAQEngine

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ISRO Recruitment RAG FAQ Chatbot",
    page_icon="🚀",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Hardcoded production defaults (no sidebar)
# ---------------------------------------------------------------------------
PERSIST_DIR = "./isro_vectordb"
RETRIEVAL_MODE = "hybrid"
TOP_K = 6
ENABLE_CACHE = True


# ---------------------------------------------------------------------------
# Chunk display cleaner
# ---------------------------------------------------------------------------
def clean_chunk_for_display(text: str) -> str:
    """Remove Devanagari, raw markdown noise, and extra whitespace from a chunk."""
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[\u0900-\u097F]+", "", text)          # Strip Devanagari
    text = re.sub(r"\*{1,3}", "", text)                    # Strip bold/italic markers
    text = re.sub(r"(?<!\|)\|(?!\|)", " | ", text)        # Normalize pipe characters
    text = re.sub(r"[ \t]{2,}", " ", text)                 # Collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)                 # Collapse excess blank lines
    # Remove lines that are just dots, slashes, or punctuation (bilingual noise)
    lines = [ln for ln in text.splitlines() if re.search(r"[A-Za-z0-9]", ln)]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Engine initialisation (cached across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_engine() -> ISROFAQEngine:
    return ISROFAQEngine(
        persist_dir=PERSIST_DIR,
        mode=RETRIEVAL_MODE,
        k=TOP_K,
        embedding_provider="huggingface",
        embedding_model="all-MiniLM-L6-v2",
        enable_cache=ENABLE_CACHE,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🚀 ISRO Recruitment FAQ Chatbot")
st.caption(
    "AI-powered Q&A grounded in official ISRO / ICRB recruitment notifications. "
    "All answers cite the source advertisement and section."
)
st.divider()

# ---------------------------------------------------------------------------
# Vector DB guard
# ---------------------------------------------------------------------------
if not os.path.exists(PERSIST_DIR):
    st.error(
        "⚠️ Vector database not found. "
        "Please run `python src/ingest.py` first to index the notification PDFs."
    )
    st.stop()

engine = get_engine()

# ---------------------------------------------------------------------------
# Chat history initialisation
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Render existing messages
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "meta" in msg and msg["meta"]:
            meta = msg["meta"]
            if meta.get("cached"):
                st.caption("⚡ Served from cache · 0 tokens used")
            else:
                st.caption(
                    f"🤖 {meta.get('provider', 'LLM')} · "
                    f"Model: `{meta.get('model_used', 'N/A')}`"
                )

# ---------------------------------------------------------------------------
# Sample question chips (shown only before first message)
# ---------------------------------------------------------------------------
if len(st.session_state.messages) == 0:
    st.markdown("**💡 Frequently Asked Questions — click to ask:**")
    sample_questions = [
        "What is the minimum percentage or CGPA required for Post Code BE001?",
        "What is the application fee and is it refundable?",
        "Are candidates with a BE/B.Tech in Artificial Intelligence eligible for Post Code BE003?",
    ]
    cols = st.columns(len(sample_questions))
    quick_query = None
    for col, q in zip(cols, sample_questions):
        if col.button(q, use_container_width=True):
            quick_query = q
else:
    quick_query = None

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
chat_input = st.chat_input("Ask about eligibility, post codes, age limit, fees, or exam patterns…")
user_query = chat_input or quick_query

# ---------------------------------------------------------------------------
# Query → answer pipeline
# ---------------------------------------------------------------------------
if user_query:
    # Append and render user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate and render assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching ISRO notifications and generating answer…"):
            try:
                result = engine.query(user_query)
                answer = result["answer"]
                docs = result.get("retrieved_docs", [])
                is_cached = result.get("cached", False)
                provider = result.get("provider", "LLM")
                model_used = result.get("model_used", "N/A")

                st.markdown(answer)

                # Attribution line
                if is_cached:
                    st.caption("⚡ Served from cache · 0 tokens used")
                else:
                    st.caption(f"🤖 {provider} · Model: `{model_used}`")

                # Retrieved passages expander
                if docs:
                    with st.expander(f"📚 Source passages retrieved ({len(docs)})"):
                        for i, doc in enumerate(docs, 1):
                            advt = doc.metadata.get("advt_no", "Unknown")
                            section = (
                                doc.metadata.get("H2")
                                or doc.metadata.get("H1")
                                or "General"
                            )
                            st.markdown(f"**Passage {i}** · `{advt}` — *{section}*")
                            cleaned = clean_chunk_for_display(doc.page_content)
                            preview = cleaned[:600] + ("…" if len(cleaned) > 600 else "")
                            st.code(preview, language="")
                            if i < len(docs):
                                st.divider()

                # Persist to session
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "meta": {
                            "cached": is_cached,
                            "provider": provider,
                            "model_used": model_used,
                        },
                    }
                )

            except Exception as exc:
                st.error(f"Error generating answer: {exc}")
