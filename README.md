# Ollama LLM Playground

Experiments with running LLMs locally through [Ollama](https://ollama.com), from a first API call up to a small PDF question-answering (RAG) pipeline.

## What's here

| File | What it does |
|---|---|
| `start-1.py` | Streams a response from the Ollama REST API |
| `start-2.py` | Same idea using the `ollama` Python package |
| `Modelfile` | Custom `llama3.2` model with its own system prompt and temperature |
| `categorizer.py` | Uses the model to categorize `data/grocery_list.txt` |
| `pdf-rag.py` | RAG over a PDF: load, chunk, embed into Chroma, retrieve, answer |
| `pdf-rag-refactor` | Refactored version of the RAG script |

## Setup

Requires [Ollama](https://ollama.com) and Python 3.10 or newer (current LangChain does not support older versions).

```bash
ollama pull llama3.2
ollama pull nomic-embed-text      # embeddings for the RAG scripts
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # optional: point OLLAMA_HOST at another server
python3 start-2.py
```

The RAG scripts read `data/ai_history_sample.pdf`; put any PDF there (PDFs in `data/` are not committed). They build the Chroma vector store in memory on each run, so nothing is written to disk; `chroma_db/` is ignored in case you add persistence.
