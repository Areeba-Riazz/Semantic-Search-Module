[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/15si9kMD)
# CS-4015 Agentic AI  
## Homework 1 – Phase 1: Semantic Search Module

📄 **Read the complete assignment description:**  
HW1_Phase1_AgenticAI.pdf

---

##  Objective
The goal of this assignment is to build the **memory system** of an AI Research Assistant.
You will design a **semantic search engine** that retrieves academic documents based on
meaning rather than keywords.

This phase focuses on:
- Document embeddings
- Vector databases
- Semantic retrieval
- Retrieval quality analysis

---

##  Tasks Overview

### Task 1: GUI-Based Data Selection
Your GUI must allow the user to:
- Upload or select a dataset (minimum 10–15 text documents)
- View dataset statistics (number of documents, size)
- No dataset should be hard-coded in the backend

### Task 2: Embedding & Vector Store Configuration
The GUI must allow the user to select:
- A Hugging Face embedding model
- A vector database (FAISS or Chroma)

Based on these selections:
- Generate embeddings
- Store them using LangChain

### Task 3: Semantic Retrieval
Your application must provide:
- A query input box
- A configurable `top-k` value
- Clearly ordered retrieval results based on relevance

### Task 4: Retrieval Evaluation & Analysis
Test your system with:
- Multiple queries
- Different datasets
- Different embedding models

Analyze and document the quality of retrieval results.

---

## 📁 Project Structure (Must Follow)

```
hw1-phase-1-semantic-search-module/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── .venv/                        # Virtual environment (not committed)
├── HW1_Phase1_AgenticAI.pdf      # Assignment specification
├── app/
│   ├── config.py                 # Paths, model lists, defaults
│   ├── gui.py                    # Streamlit GUI (main application)
│   └── main.py                   # CLI entry point
├── data/
│   ├── README.md
│   └── data_loader.py            # Document ingestion & chunking
├── embeddings/
│   ├── README.md
│   └── embedding_manager.py      # HuggingFace embedding wrapper
├── Vector_Store/
│   ├── README.md
│   └── vector_store_manager.py   # FAISS / Chroma creation & query
└── experiments/
    ├── README.md
    └── report/
        └── report_template.md    # Report template
```

## 📦 Deliverables

1. GUI-based application  
2. Complete source code  
3. Short report (2–3 pages)  

All deliverables must be pushed to this repository.

---

## 🚫 Restrictions
- Do NOT hard-code datasets
- Do NOT hard-code embedding models
- Do NOT hard-code vector database choice

---

## 📝 Submission Instructions
Commit your work regularly.  
Your **latest commit before the deadline** will be graded.

---

## ⚠️ Academic Integrity
This is an individual assignment.  
Plagiarism or code sharing will result in disciplinary action.

---

## 🚀 How to Run

### 1. Create a virtual environment

```bash
# From the project root directory
python -m venv .venv
```

### 2. Activate the virtual environment

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (CMD)
.\.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the application

```bash
# From the project root, with .venv activated
streamlit run app/gui.py

# Alternative: use the main.py entry point
python app/main.py
```

The app will open at `http://localhost:8501`.

---

## 🧪 How to Test

1. **Upload** at least 10–15 `.txt` documents via the file uploader in Section 1.
2. **Select** an embedding model and vector store in Section 2, then click **Build Index**.
3. **Enter** a natural-language query in Section 3, set your desired **Top-K**, and click **Search**.
4. **Review** the ranked results displayed as cards with relevance scores.
5. **Check** the Evaluation panel (Section 4) for a log of all queries run in the session.

---

## 🔄 How to Switch Models / Vector DBs

- In **Section 2** of the GUI, change the dropdown selections:
  - **Embedding Model:** choose any of the listed Hugging Face models.
  - **Vector Database:** switch between FAISS and Chroma.
- Click **Build Index** again to rebuild with the new configuration.
- Re-run your queries in Section 3 and compare results in Section 4.
