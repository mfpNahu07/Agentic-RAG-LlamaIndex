import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import Document
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
import fitz  # PyMuPDF - handles ONLYOFFICE/complex PDF formats correctly

load_dotenv()

def load_docs_from_dir(directory: str) -> list[Document]:
    """Load all files from a directory, using PyMuPDF for PDFs."""
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.lower().endswith(".pdf"):
            # Use PyMuPDF for reliable text extraction from any PDF type
            pdf = fitz.open(filepath)
            text = "\n".join(page.get_text() for page in pdf)
            pdf.close()
            if text.strip():
                documents.append(Document(text=text, metadata={"filename": filename}))
                print(f"  [OK] PDF parsed: {filename} ({len(text)} chars)")
            else:
                print(f"  [WARN] No text extracted from: {filename}")
        elif filename.lower().endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append(Document(text=text, metadata={"filename": filename}))
            print(f"  [OK] TXT loaded: {filename} ({len(text)} chars)")
    return documents

def build_agentic_indices():
    print("[*] Configuring GoogleGenAI Embedding model...")
    Settings.embed_model = GoogleGenAIEmbedding(model_name="models/gemini-embedding-001")
    Settings.llm = None

    # 1. Process n8n documentation
    print("[*] Processing n8n documentation...")
    if os.path.exists("docs/n8n") and os.listdir("docs/n8n"):
        n8n_docs = load_docs_from_dir("docs/n8n")
        n8n_index = VectorStoreIndex.from_documents(n8n_docs)
        n8n_index.storage_context.persist(persist_dir="storage/n8n")
        print(f"[OK] n8n index saved ({len(n8n_docs)} documents)")

    # 2. Process Claude Code documentation
    print("[*] Processing Claude Code documentation...")
    if os.path.exists("docs/claude") and os.listdir("docs/claude"):
        claude_docs = load_docs_from_dir("docs/claude")
        claude_index = VectorStoreIndex.from_documents(claude_docs)
        claude_index.storage_context.persist(persist_dir="storage/claude")
        print(f"[OK] Claude index saved ({len(claude_docs)} documents)")

    print("[DONE] Data ingestion pipeline completed successfully!")

if __name__ == "__main__":
    build_agentic_indices()
