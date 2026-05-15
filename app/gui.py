"""
GUI Module — Streamlit-based professional academic interface for the
AI Research Assistant Phase-1: Semantic Search Module.

Sections:
  1. Sidebar — branding & navigation
  2. Dataset Panel — upload documents, view stats
  3. Configuration Panel — select embedding model & vector DB
  4. Query Panel — enter query, set top-k
  5. Results Panel — ranked card-based results
  6. Evaluation Panel — query history & comparison
"""

import sys
import os
import time
import datetime

# ── Ensure project root is on sys.path ──────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from app.config import (
    DATA_DIR,
    VECTOR_STORE_DIR,
    EMBEDDING_MODELS,
    VECTOR_DB_OPTIONS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VECTOR_DB,
    DEFAULT_TOP_K,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from data.data_loader import (
    load_documents_from_uploaded_files,
    get_dataset_stats,
    chunk_documents,
)
from embeddings.embedding_manager import get_embedding_model, get_available_models
from Vector_Store.vector_store_manager import create_vector_store, query_vector_store


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="AI Research Assistant — Semantic Search",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Custom CSS for a professional, academic-grade look
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown(
    """
    <style>
    /* ── Google Font ───────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Root Variables (dark-mode native) ─────── */
    :root {
        --accent:       #5B9CF5;
        --accent-soft:  #3E6FBF;
        --accent-glow:  rgba(91,156,245,0.15);
        --card-bg:      rgba(255,255,255,0.05);
        --card-border:  rgba(255,255,255,0.10);
        --card-hover:   rgba(255,255,255,0.08);
        --header-from:  #1E3A5F;
        --header-to:    #264D73;
        --text-primary: #E8ECF1;
        --text-secondary:#A0AEBD;
        --success:      #4ADE80;
        --warning:      #FBBF24;
        --error:        #F87171;
    }

    /* ── Sidebar Styling ──────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F1B2D 0%, #162236 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] * {
        color: #C5D0DC !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }

    /* ── Section Headers ──────────────────────── */
    .section-header {
        background: linear-gradient(135deg, var(--header-from), var(--header-to));
        color: #FFFFFF;
        padding: 14px 22px;
        border-radius: 10px;
        margin-bottom: 16px;
        font-weight: 600;
        font-size: 1.05rem;
        letter-spacing: 0.3px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* ── Result Card ──────────────────────────── */
    .result-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-left: 4px solid var(--accent);
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 14px;
        transition: background 0.2s ease, box-shadow 0.2s ease;
    }
    .result-card:hover {
        background: var(--card-hover);
        box-shadow: 0 4px 24px rgba(91,156,245,0.10);
    }
    .result-rank {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, var(--accent), var(--accent-soft));
        color: #FFFFFF;
        min-width: 28px;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
        margin-right: 10px;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 8px rgba(91,156,245,0.25);
    }
    .result-score {
        display: inline-block;
        background: var(--accent-glow);
        color: var(--accent);
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .result-source {
        color: var(--text-secondary);
        font-size: 0.8rem;
        margin-top: 4px;
    }
    .result-content {
        color: var(--text-primary);
        font-size: 0.92rem;
        margin-top: 8px;
        line-height: 1.55;
    }

    /* ── Stat Cards ───────────────────────────── */
    .stat-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .stat-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: var(--accent);
    }
    .stat-label {
        font-size: 0.82rem;
        color: var(--text-secondary);
        margin-top: 2px;
    }

    /* ── Status Badge ─────────────────────────── */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-ready   { background: rgba(74,222,128,0.15); color: #4ADE80; }
    .status-warning { background: rgba(251,191,36,0.15); color: #FBBF24; }
    .status-error   { background: rgba(248,113,113,0.15); color: #F87171; }

    /* ── Query Log Table ──────────────────────── */
    .log-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    .log-table th {
        background: var(--header-from);
        color: #FFFFFF;
        padding: 10px 14px;
        text-align: left;
        border-bottom: 2px solid var(--accent-soft);
    }
    .log-table td {
        padding: 8px 14px;
        border-bottom: 1px solid var(--card-border);
        color: var(--text-primary);
    }
    .log-table tr:nth-child(even) { background: rgba(255,255,255,0.03); }
    .log-table tr:hover { background: rgba(255,255,255,0.06); }

    /* ── Hide default Streamlit decoration ─── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Divider ───────────────────────────── */
    .custom-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 28px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Session State Defaults
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
defaults = {
    "documents": [],
    "chunks": [],
    "vector_store": None,
    "index_built": False,
    "current_model": None,
    "current_db": None,
    "query_log": [],      # list of dicts for evaluation
    "build_time": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar — Branding & Status
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("## 🔬 AI Research Assistant")
    st.markdown("##### Phase 1 — Semantic Search Module")
    st.markdown("---")

    # ── System Status ──
    st.markdown("### 📊 System Status")

    docs_loaded = len(st.session_state.documents) > 0
    index_ready = st.session_state.index_built

    if docs_loaded:
        st.markdown(
            '<span class="status-badge status-ready">✓ Documents Loaded</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge status-warning">⚠ No Documents</span>',
            unsafe_allow_html=True,
        )

    if index_ready:
        st.markdown(
            '<span class="status-badge status-ready">✓ Index Ready</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Model:** {st.session_state.current_model}")
        st.markdown(f"**Store:** {st.session_state.current_db}")
    else:
        st.markdown(
            '<span class="status-badge status-warning">⚠ Index Not Built</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem; color:#A0AEC0;'>"
        "CS-4015 Agentic AI · HW1"
        "</p>",
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Title
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown(
    """
    <div style="text-align:center; padding: 10px 0 24px 0;">
        <h1 style="color:#E8ECF1; font-weight:700; margin-bottom:4px;">
            🔬 Semantic Search Engine
        </h1>
        <p style="color:#A0AEBD; font-size:1.05rem;">
            Upload academic documents · Build vector indices · Retrieve by meaning
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1 — Dataset Upload
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<div class="section-header">📁 1 · Dataset Upload</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload text documents (.txt)",
    type=["txt"],
    accept_multiple_files=True,
    help="Upload at least 10–15 text documents. No datasets are hard-coded.",
    key="file_uploader",
)

if uploaded_files:
    # Only reload when the set of uploaded files actually changes
    current_file_names = sorted([f.name for f in uploaded_files])
    prev_file_names = st.session_state.get("_uploaded_file_names", [])

    if current_file_names != prev_file_names:
        with st.spinner("Loading documents …"):
            documents = load_documents_from_uploaded_files(uploaded_files, DATA_DIR)
            st.session_state.documents = documents
            st.session_state._uploaded_file_names = current_file_names
            # reset index because documents changed
            st.session_state.index_built = False
            st.session_state.vector_store = None

    stats = get_dataset_stats(st.session_state.documents)

    # Stat cards row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="stat-card"><div class="stat-value">{stats["num_documents"]}</div>'
            f'<div class="stat-label">Documents</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        size_kb = stats["total_size_bytes"] / 1024
        st.markdown(
            f'<div class="stat-card"><div class="stat-value">{size_kb:.1f} KB</div>'
            f'<div class="stat-label">Total Size</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="stat-card"><div class="stat-value">{stats["avg_document_length"]}</div>'
            f'<div class="stat-label">Avg Characters</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="stat-card"><div class="stat-value">{stats["total_characters"]}</div>'
            f'<div class="stat-label">Total Characters</div></div>',
            unsafe_allow_html=True,
        )

    # Document list
    with st.expander("📄 View uploaded documents", expanded=False):
        for doc in st.session_state.documents:
            st.markdown(
                f"**{doc.metadata['source']}** — "
                f"{doc.metadata['size_bytes']:,} bytes"
            )
else:
    st.info("👆 Upload your text documents above to get started.")


st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2 — Embedding & Vector Store Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown(
    '<div class="section-header">⚙️ 2 · Embedding & Vector Store Configuration</div>',
    unsafe_allow_html=True,
)

col_cfg1, col_cfg2 = st.columns(2)

with col_cfg1:
    selected_model = st.selectbox(
        "Hugging Face Embedding Model",
        options=get_available_models(),
        index=get_available_models().index(DEFAULT_EMBEDDING_MODEL),
        help="Choose the sentence-transformer model for generating embeddings.",
    )

with col_cfg2:
    selected_db = st.selectbox(
        "Vector Database",
        options=VECTOR_DB_OPTIONS,
        index=VECTOR_DB_OPTIONS.index(DEFAULT_VECTOR_DB),
        help="Choose between FAISS (fast, in-memory) or Chroma (persistent, metadata-rich).",
    )

col_adv1, col_adv2 = st.columns(2)
with col_adv1:
    chunk_size = st.slider(
        "Chunk Size (characters)",
        min_value=100,
        max_value=2000,
        value=DEFAULT_CHUNK_SIZE,
        step=50,
        help="Maximum characters per text chunk.",
    )
with col_adv2:
    chunk_overlap = st.slider(
        "Chunk Overlap (characters)",
        min_value=0,
        max_value=500,
        value=DEFAULT_CHUNK_OVERLAP,
        step=10,
        help="Overlap between consecutive chunks for context continuity.",
    )

# Build Index button
if st.button("🚀 Build Index", use_container_width=True, type="primary"):
    if not st.session_state.documents:
        st.error("⚠️ Please upload documents first before building the index.")
    else:
        with st.spinner("Chunking documents …"):
            chunks = chunk_documents(
                st.session_state.documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            st.session_state.chunks = chunks

        st.info(f"✂️ Created **{len(chunks)}** chunks from {len(st.session_state.documents)} documents.")

        with st.spinner(f"Loading embedding model **{selected_model}** …"):
            emb_model = get_embedding_model(selected_model)

        with st.spinner(f"Building **{selected_db}** vector store …"):
            persist_dir = os.path.join(VECTOR_STORE_DIR, f"{selected_db.lower()}_store")
            vs, build_time = create_vector_store(
                chunks, emb_model, store_type=selected_db, persist_directory=persist_dir
            )
            st.session_state.vector_store = vs
            st.session_state.index_built = True
            st.session_state.current_model = selected_model
            st.session_state.current_db = selected_db
            st.session_state.build_time = build_time

        st.success(
            f"✅ Index built successfully!  \n"
            f"**Model:** {selected_model} · **Store:** {selected_db} · "
            f"**Chunks:** {len(chunks)} · **Time:** {build_time:.2f}s"
        )


st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3 — Semantic Query
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown(
    '<div class="section-header">🔍 3 · Semantic Query</div>',
    unsafe_allow_html=True,
)

query_col, topk_col = st.columns([3, 1])
with query_col:
    query_text = st.text_input(
        "Enter your search query",
        placeholder="e.g.  What are the effects of climate change on agriculture?",
        help="Type a natural-language query. The system retrieves documents by semantic meaning.",
    )
with topk_col:
    top_k = st.slider("Top-K", min_value=1, max_value=20, value=DEFAULT_TOP_K)

if st.button("🔎 Search", use_container_width=True, type="primary"):
    if not query_text.strip():
        st.warning("Please enter a query.")
    elif not st.session_state.index_built:
        st.error("⚠️ Build the index first (Section 2) before running a query.")
    else:
        with st.spinner("Searching …"):
            search_start = time.time()
            results = query_vector_store(
                st.session_state.vector_store, query_text, top_k=top_k
            )
            search_time = time.time() - search_start

        # ── Log for evaluation ──
        st.session_state.query_log.append(
            {
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "query": query_text,
                "top_k": top_k,
                "model": st.session_state.current_model,
                "db": st.session_state.current_db,
                "num_results": len(results),
                "search_time": f"{search_time:.3f}s",
            }
        )

        # ── Display Results ──────────────────────────
        st.markdown(
            f"<p style='color:#A0AEBD; font-size:0.9rem;'>"
            f"Found <b>{len(results)}</b> results in <b>{search_time:.3f}s</b> "
            f"&nbsp;·&nbsp; <span style='font-size:0.82rem;'>ℹ️ Lower score = more relevant (distance-based ranking)</span></p>",
            unsafe_allow_html=True,
        )

        for rank, (doc, score) in enumerate(results, start=1):
            # Both FAISS and Chroma return L2 distance scores via LangChain.
            # Lower score = more semantically similar to the query.
            source = doc.metadata.get("source", "Unknown")
            content_preview = doc.page_content[:500]

            st.markdown(
                f"""
                <div class="result-card">
                    <div>
                        <span class="result-rank">#{rank}</span>
                        <span class="result-score">Score: {score:.4f}</span>
                        <span class="result-source" style="margin-left:12px;">
                            📄 {source}
                        </span>
                    </div>
                    <div class="result-content">{content_preview}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4 — Evaluation & Query History
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown(
    '<div class="section-header">📈 4 · Evaluation & Query History</div>',
    unsafe_allow_html=True,
)

if st.session_state.query_log:
    st.markdown(
        "Use this log to compare retrieval results across **different models**, "
        "**vector stores**, and **queries** for your report."
    )

    # Build HTML table
    rows = ""
    for entry in reversed(st.session_state.query_log):
        rows += (
            f"<tr>"
            f"<td>{entry['timestamp']}</td>"
            f"<td>{entry['query'][:60]}{'…' if len(entry['query'])>60 else ''}</td>"
            f"<td>{entry['model']}</td>"
            f"<td>{entry['db']}</td>"
            f"<td>{entry['top_k']}</td>"
            f"<td>{entry['num_results']}</td>"
            f"<td>{entry['search_time']}</td>"
            f"</tr>"
        )

    st.markdown(
        f"""
        <table class="log-table">
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Query</th>
                    <th>Model</th>
                    <th>DB</th>
                    <th>K</th>
                    <th>Results</th>
                    <th>Latency</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🗑️ Clear Query Log"):
        st.session_state.query_log = []
        st.rerun()
else:
    st.info("Run queries above to build an evaluation log here.")
