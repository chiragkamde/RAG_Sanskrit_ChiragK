# Sanskrit Document Retrieval-Augmented Generation (RAG) System

**Project Title:** Sanskrit Document Retrieval-Augmented Generation (RAG) System

---

## Objective

The objective of this assignment is to design and implement a **Retrieval-Augmented Generation (RAG) system** capable of processing and answering queries based on Sanskrit documents. The system operates fully on **CPU-based inference** (no GPU required).

---

## Description

This project is an end-to-end RAG pipeline that:

1. **Ingests** Sanskrit documents in **.txt**, **.pdf**, or **.docx** format (from a chosen domain).
2. **Preprocesses and indexes** these documents for efficient retrieval (chunking, embeddings, FAISS).
3. **Provides a query interface** that accepts user input in **Sanskrit or transliterated text**.
4. **Retrieves** relevant context chunks from the indexed corpus (top-k semantic search).
5. **Generates** coherent responses using a **CPU-based Large Language Model (LLM)** integrated into the pipeline.

The RAG architecture follows standard practices and is **modular** (document loader, preprocessing, embedding/index, retriever, generator).

---

## Requirements (Assignment Compliance)

- **Implementation:** Python.
- **Libraries:** HuggingFace Transformers, FAISS, sentence-transformers; Streamlit for the query interface. (LangChain-style modular pipeline without LangChain dependency.)
- **CPU-only:** No GPU acceleration; all inference runs on CPU.
- **Multiple documents:** Supports multiple documents and scales via FAISS index and chunked ingestion.
- **Documentation:** This README provides setup and usage instructions.

---

## Architecture, Design Choices, and Usage (Brief Report)

### Architecture Overview

The pipeline is modular and follows standard RAG stages:

```
Document Loader (.txt / .pdf / .docx)
    → Preprocessing & Chunking (clean text, 500 tokens, 50 overlap)
    → Embedding Generation (paraphrase-multilingual-MiniLM-L12-v2)
    → Vector Index (FAISS)
    → Retriever (top-k = 3)
    → LLM Generator (flan-t5-base, CPU; optional Ollama-served model if available)
    → Answer
```

- **Document Loader:** Reads Sanskrit (or English) documents from `.txt`, `.pdf`, and `.docx` files. Supports both on-disk corpus and **uploaded files** in the web UI.
- **Preprocessing:** Cleans text and chunks with configurable size/overlap (default 500 tokens, 50 overlap).
- **Embeddings:** Multilingual sentence embeddings (MiniLM) for Sanskrit and transliterated text; CPU-only.
- **Vector Store:** FAISS index for fast similarity search; index can be built from corpus directory or from uploaded documents.
- **Retriever:** Returns top-k relevant chunks (default k=3).
- **Generator:** flan-t5-base on CPU for answer generation. An Ollama-hosted model (e.g. `llama2-13b` or other supported LLM) can be selected via the UI sidebar if you have Ollama installed and a model pulled locally.
- **Query Interface:** Web UI (Streamlit) and CLI; accepts Sanskrit or transliterated queries.

### Design Choices

- **CPU-only:** All models (embeddings + LLM) run on CPU for portability and assignment compliance.
- **Modularity:** Clear separation of loader, preprocessor, embedding/index, retriever, and generator for maintainability.
- **Multiple documents:** Corpus can be `nalanda_library/` (or `vedic_texts/`), or user-uploaded PDF/TXT/DOCX in the UI.
- **Sanskrit + transliteration:** Multilingual embedding model and flexible query input support both.

### File Organization (No Code Changes; Names and Directories as in Project)

```
VedicRAG_AI-main/
├── README.md                           # This file (project title & documentation)
├── Data Scrapping/                     # Main application folder
│   ├── requirements.txt                # Python dependencies
│   ├── simple_rag_demo.py              # RAG pipeline (loader, preprocessing, FAISS, retriever, generator)
│   ├── ollama_rag_ui.py                # Streamlit query interface (upload + Q&A)
│   └── VedicDatasetGenerator.py       # Builds sample corpus (nalanda_library) if needed
├── nalanda_library/                    # Default Sanskrit document corpus (sample/domain data)
│   ├── dataset_metadata.json
│   ├── nalanda_corpus_part_*.txt
│   └── faiss_index/                    # FAISS index (built on first run or via CLI)
├── dharmaganj/                         # Optional additional corpus
└── Docker Code/                        # Optional Docker deployment
    ├── docker-compose.yml
    ├── Dockerfile
    ├── main.py
    └── ingest.py
```

---

## Setup and Usage Instructions

### Prerequisites

- Python 3.9+
- CPU-only environment (no GPU required)

### 1. Install Dependencies

```bash
cd "Data Scrapping"
pip install -r requirements.txt
```

### 2. (Optional) Build Sample Corpus

If you want to use the included Sanskrit corpus instead of (or in addition to) uploads:

```bash
cd "Data Scrapping"
python VedicDatasetGenerator.py
```

This creates or populates `nalanda_library/` with sample documents. You can also place your own `.txt` or `.pdf` (and optionally `.docx`) files in `nalanda_library/` and re-run the app to index them.

### 3. Run the Query Interface (Web UI)

```bash
cd "Data Scrapping"
streamlit run ollama_rag_ui.py
```

- Open the URL shown (e.g. `http://localhost:8501`).
- **Option A — Use your own documents:** Upload **.pdf**, **.txt**, or **.docx** files (Sanskrit or English), check **“Use uploaded documents for Q&A”**, then ask questions. Answers are based only on uploaded files.
- **Option B — Use default corpus:** Leave uploads unchecked; the app uses the indexed `nalanda_library/` (run step 2 first if the folder is empty).

Queries can be in **Sanskrit or transliterated text**. The system retrieves relevant chunks and generates answers using the CPU-based LLM (flan-t5-base by default; optional Ollama in sidebar).

> **Optional Ollama LLM Backend**
>
> The sidebar of the web interface allows you to switch from the built-in `flan-t5-base` model to an Ollama‑served model. This is purely optional and requires an [Ollama](https://ollama.com) installation with a compatible LLM pulled locally. Recommended models include `llama2-13b`, `llama2-70b`, or any GPT‑style model supported by Ollama. Before starting Streamlit:
>
> 1. Install Ollama (see [installation instructions](https://ollama.com/docs/installation)).
> 2. Run `ollama pull llama2-13b` (or your chosen model) in a terminal.
> 3. Start the Ollama daemon if it isn’t already running: `ollama run llama2-13b`.
>
> Once the model is available, select **"Use Ollama model"** in the UI sidebar. The interface will send generation requests to `localhost:11434` by default. If you don’t have Ollama or don’t select an Ollama model, the system will fall back to `flan-t5-base` on CPU.

### 4. (Optional) CLI

```bash
cd "Data Scrapping"
python simple_rag_demo.py "your question in Sanskrit or English"
```

---

## Sample Sanskrit Documents for Testing

- The **nalanda_library/** directory (after running `VedicDatasetGenerator.py`) contains sample Sanskrit/English verse data in structured `.txt` format (e.g. Itihasa corpus).
- You can add your own **.txt**, **.pdf**, or **.docx** files under `nalanda_library/` or use the **web UI upload** for ad-hoc documents. No code changes are required; only document names and directory layout as above.

---

## Deliverables (Summary)

| Deliverable                   | Location / Description                                                                                        |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Source code**               | `Data Scrapping/simple_rag_demo.py`, `ollama_rag_ui.py`, `VedicDatasetGenerator.py`; optional `Docker Code/`. |
| **Sample Sanskrit documents** | `nalanda_library/` (after running `VedicDatasetGenerator.py`); optional `dharmaganj/`.                        |
| **Brief report**              | This README: architecture, design choices, setup and usage instructions.                                      |
| **Demo video**                | Optional; can showcase upload, Sanskrit/transliterated query, and generated answer.                           |

---

## Evaluation Criteria (Self-Check)

- **Correctness and completeness:** Pipeline ingests .txt/.pdf/.docx, preprocesses, indexes with FAISS, retrieves top-k chunks, and generates answers with a CPU-based LLM.
- **Efficiency and scalability:** FAISS and chunked processing support multiple documents; CPU-only inference.
- **Quality of responses:** Answers are grounded in retrieved chunks; prompt format encourages Sanskrit or English explanation.
- **Code readability and documentation:** Modular components and this README with clear setup and usage.
- **Innovation / bonus:** Upload of PDF/TXT/DOCX in the web UI and querying over uploaded documents without changing code; support for Sanskrit and transliterated queries.

---

## License and Acknowledgments

This project is for educational and assignment purposes. Sanskrit and Indic text sources are used in accordance with their respective terms. No code has been modified for this README; only project title, documentation, and directory/file names are aligned with the assignment objective.
