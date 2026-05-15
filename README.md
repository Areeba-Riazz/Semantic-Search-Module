# Semantic Search Module
**CS-4015: Agentic AI**  
**National University of Computer & Emerging Sciences (FAST-NUCES)**

**Student Name:** Areeba Riaz  
**Student ID:** 22i-1244

---

##  Project Overview

This project implements the **Memory System** of an AI Research Assistant — a Semantic Search Module that retrieves academic documents based on **contextual meaning** rather than keyword matching.

Unlike traditional keyword search, this system understands the *intent* behind a query. A search for *"how machines learn from data"* correctly surfaces documents about supervised learning and neural networks, even if those exact words never appear in the text.

This is Phase 1 of a multi-part Agentic AI pipeline, laying the foundation for a full **Retrieval-Augmented Generation (RAG)** system.

---

##  System Architecture

The system follows a modular, sequential pipeline where each layer has a single responsibility.

### 1. Data Module — `data/data_loader.py`
- Handles dynamic file uploads through the Streamlit GUI — no datasets are hard-coded
- Implements a **recursive character text splitter** using paragraph, sentence, and word boundaries
- Default chunk size of **500 characters** with **50 character overlap** to preserve semantic continuity across boundaries
- Saves uploaded files to disk and wraps them as LangChain `Document` objects with metadata

### 2. Embeddings Module — `embeddings/embedding_manager.py`
- Wraps **5 HuggingFace sentence-transformer models** via `langchain-huggingface`
- Model selection is fully dynamic — chosen at runtime through the GUI
- All embeddings are **L2-normalized** for consistent cosine similarity scoring
- Runs on CPU by default, compatible with any machine without a GPU

### 3. Vector Store Module — `Vector_Store/vector_store_manager.py`
- Implements **FAISS** for high-speed in-memory similarity search
- Implements **ChromaDB** for persistent, metadata-rich document storage
- Both backends are created, persisted, and queried through a unified interface
- Stale indices are automatically cleared before each rebuild to prevent stale data

### 4. GUI Module — `app/gui.py`
- Built with **Streamlit** — professional dark-mode dashboard with glassmorphism styling
- Four clearly separated sections: Upload → Configure → Query → Evaluate
- Real-time feedback: chunk count, index build time, search latency
- Query history log for comparing performance across models and databases

---

## Embedding Models

| Model | Dimensions | Characteristics |
|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Lightweight, very fast, good general-purpose baseline |
| `all-mpnet-base-v2` | 768 | Higher accuracy, slower load, best for complex queries |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | Optimized specifically for question-answering retrieval |
| `paraphrase-MiniLM-L6-v2` | 384 | Strong on paraphrase detection and semantic similarity |
| `all-distilroberta-v1` | 768 | Robust general performance with RoBERTa backbone |

> **Score interpretation:** Both FAISS and Chroma return L2 distance scores — **lower score = more semantically relevant**.

---

## Installation & Usage

**Prerequisites:** Python 3.10+

**1. Clone and set up a virtual environment**
```bash
git clone https://github.com/areeba-riazz/semantic-search-module.git
cd semantic-search-module
python -m venv .venv
```

**2. Activate the environment**
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

**4. Launch the app**
```bash
streamlit run app/gui.py
```
Opens at `http://localhost:8501`

**How to use:**
1. **Upload** at least 10–15 `.txt` documents via the file uploader
2. **Select** an embedding model and vector store (FAISS or Chroma)
3. **Configure** chunk size and overlap using the sliders
4. **Click Build Index** to generate the vector store
5. **Enter a query** and set Top-K, then click Search
6. **Review** ranked results with relevance scores in the Evaluation log

---

## Repository Structure

```
semantic-search-module/
├── app/
│   ├── config.py               # Paths, model registry, default parameters
│   ├── gui.py                  # Streamlit dashboard (main application)
│   └── main.py                 # CLI entry point with dependency check
├── data/
│   ├── data_loader.py          # Document ingestion, stats, and chunking
│   └── README.md
├── embeddings/
│   ├── embedding_manager.py    # HuggingFace embedding model wrapper
│   └── README.md
├── Vector_Store/
│   ├── vector_store_manager.py # FAISS / Chroma creation, persistence & query
│   └── README.md
├── experiments/
│   ├── README.md
│   └── report/
│       ├── Phase1_Report.docx  # Full experiment report
│       └── report_template.md
├── .gitignore
├── requirements.txt
└── HW1_Phase1_AgenticAI.pdf    # Assignment specification
```

---

*Developed for academic purposes as part of CS-4015 Agentic AI at FAST-NUCES.*
