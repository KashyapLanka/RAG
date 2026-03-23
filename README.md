# RAG (Retrieval-Augmented Generation) System

A comprehensive Retrieval-Augmented Generation system that combines document retrieval with local LLM inference. This project includes Python scripts for document processing and vector database management.

## Project Overview

This RAG system demonstrates how to:
- Load and process documents (PDFs) into a vector database
- Chunk documents into manageable segments for embedding
- Store and retrieve document chunks using semantic similarity
- Augment prompts with retrieved context before sending to an LLM

## Project Structure

```
RAG/
├── rag.py                   # Core RAG script for document processing and chunking
├── requirements.txt         # Python dependencies
├── chroma_db/               # Vector database storage (ChromaDB)
└── Docs/                    # Document source files
    ├── data.txt
    ├── potato_disease_literature_review.csv
    └── review paper.pdf
```

## Components

### 1. RAG Script (`rag.py`)

Core Python script that handles document ingestion, chunking, and vectorization.

**Features:**
- Document loading with `PyPDFDirectoryLoader`
- Text chunking with `RecursiveCharacterTextSplitter`
- Persistent vector storage in ChromaDB

### 2. ChromaDB Vector Database (`chroma_db/`)

Persistent vector store for embeddings and document chunks. Created automatically when `rag.py` runs.

### 3. Requirements (`requirements.txt`)

Lists Python dependencies. Install with:

```bash
pip install -r requirements.txt
```

### 4. Docs Directory (`Docs/`)

Contains source documents for RAG:
- `data.txt`
- `potato_disease_literature_review.csv`
- `review paper.pdf`

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

## Usage

```bash
python rag.py
```

This loads documents from `Docs/`, chunks text, generates embeddings, and stores vectors in `chroma_db/`.

## Notes

- `chroma_db/` is auto-created by the script.
- Add new documents in `Docs/` and re-run `rag.py`.
- Excludes `test.ipynb` from this readme scope as requested.

## Troubleshooting

- If ChromaDB import fails, ensure `chromadb` is installed.
- If `Docs/` loading fails, verify files exist and formats are supported.


## Author

Kashyap Lanka
