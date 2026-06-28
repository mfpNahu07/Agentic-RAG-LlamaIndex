# 🤖 Agentic RAG with LlamaIndex & Gemini

A multi-document reasoning chatbot that uses an autonomous **ReAct agent** to cross-reference technical documentation in real time. Built with LlamaIndex, Google Gemini, and Streamlit.

---

## What It Does

Instead of running a single vector search per query, this project implements a full **agentic loop**: the agent decides which documents to query, calls the appropriate tools, synthesizes information across sources, and produces a grounded answer — all autonomously.

**Demo scenario:** The agent can cross-reference n8n automation documentation against Claude Code technical docs to answer comparative questions like *"How does n8n's trigger-action model compare to Claude Code's agentic loop?"*

---

## Architecture

```
User Query
    │
    ▼
ReActAgent (LlamaIndex 0.14 Workflow-based)
    │
    ├── n8n_documentation_tool      ──► VectorStoreIndex (storage/n8n)
    └── claude_code_documentation_tool ──► VectorStoreIndex (storage/claude)
                                              │
                                         PyMuPDF + GoogleGenAI Embeddings
                                         (gemini-embedding-001)
```

The agent uses **Gemini 2.5 Flash** as its reasoning backbone and **gemini-embedding-001** for semantic vector search.

---

## Stack

| Layer | Technology |
|---|---|
| Agent framework | LlamaIndex Core 0.14 (`ReActAgent`) |
| LLM | Google Gemini 2.5 Flash |
| Embeddings | Google Gemini Embedding 001 |
| Vector store | LlamaIndex in-memory + disk persistence |
| PDF parsing | PyMuPDF (`fitz`) |
| UI | Streamlit |

---

## Project Structure

```
├── app.py          # Streamlit UI + agent initialization
├── ingest.py       # Document ingestion pipeline (PDF → vector index)
├── docs/
│   ├── n8n/        # Drop your n8n documentation files here
│   └── claude/     # Drop your Claude Code documentation files here
├── storage/        # Persisted vector indices (git-ignored)
├── requirements.txt
└── .env            # GOOGLE_API_KEY (git-ignored)
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/mfpNahu07/Agentic-RAG-LlamaIndex.git
cd Agentic-RAG-LlamaIndex
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate  # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_api_key_here
```


### 4. Add your documents

Drop your files into `docs/n8n/` and `docs/claude/`. Both `.txt` and `.pdf` formats are supported.

### 5. Build the vector indices

```bash
python ingest.py
```

### 6. Run the app

```bash
python -m streamlit run app.py
```

---

## How to Use

Ask any question that requires reasoning across both documents. Examples:

- *"What is the agentic loop in Claude Code and what are its steps?"*
- *"Compare the automation approach of n8n with Claude Code's tool execution model."*
- *"Which platform gives more control to the user when executing actions, and why?"*

The agent will call one or both documentation tools, retrieve the relevant chunks, and synthesize a grounded answer.

---

## Adding More Documents

To index additional sources:

1. Create a new folder under `docs/` (e.g., `docs/langchain/`)
2. Add a new `QueryEngineTool` in `app.py` pointing to the new index
3. Add a new indexing block in `ingest.py`
4. Re-run `python ingest.py`

---

## License

MIT
