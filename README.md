[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/15si9kMD)

# Semantic Search Module
### CS-4015 Agentic AI — HW1 Phase 1

A modular semantic search engine designed as the memory layer for an AI Research Assistant. Upload documents, choose an embedding model, build a vector index, and retrieve results by meaning — not keywords.

---

## Features

- **Dynamic document upload** — no hard-coded datasets
- **5 Hugging Face embedding models** — swappable at runtime
- **Dual vector store support** — FAISS and Chroma via LangChain
- **Configurable chunking** — adjustable chunk size and overlap
- **Ranked semantic retrieval** — Top-K results with relevance scores
- **Query evaluation log** — tracks latency, model, and DB across queries
- **Professional Streamlit UI** — dark-mode dashboard with stat cards

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Embeddings | HuggingFace `sentence-transformers` via `langchain-huggingface` |
| Vector Store | FAISS / Chroma via `langchain-community` |
| Chunking | Custom recursive text splitter |
| Language | Python 3.10+ |

---

## Project Structure

```
semantic-search-module/
├── app/
│   ├── config.py               # Paths, model list, defaults
│   ├── gui.py                  # Streamlit app (main UI)
│   └── main.py                 # CLI entry point
├── data/
│   ├── data_loader.py          # Document ingestion & chunking
│   └── README.md
├── embeddings/
│   ├── embedding_manager.py    # HuggingFace embedding wrapper
│   └── README.md
├── Vector_Store/
│   ├── vector_store_manager.py # FAISS / Chroma creation & query
│   └── README.md
├── experiments/
│   ├── README.md
│   └── report/
│       ├── Phase1_Report.docx
│       └── report_template.md
├── .gitignore
├── requirements.txt
└── HW1_Phase1_AgenticAI.pdf
```

---

## Getting Started

**1. Clone the repo and create a virtual environment**
```bash
python -m venv .venv
```

**2. Activate it**
```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app/gui.py
```
Opens at `http://localhost:8501`

---

## Usage

1. **Upload** `.txt` documents via the file uploader (Section 1)
2. **Select** an embedding model and vector store, then click **Build Index** (Section 2)
3. **Enter** a natural-language query and set Top-K, then click **Search** (Section 3)
4. **Review** ranked results with relevance scores (Section 3)
5. **Compare** query performance across models and DBs in the evaluation log (Section 4)

---

## Available Embedding Models

| Model | Dimensions |
|---|---|
| `all-MiniLM-L6-v2` | 384 |
| `all-mpnet-base-v2` | 768 |
| `multi-qa-MiniLM-L6-cos-v1` | 384 |
| `paraphrase-MiniLM-L6-v2` | 384 |
| `all-distilroberta-v1` | 768 |
