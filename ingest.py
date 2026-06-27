import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings

# Load environment variables
load_dotenv()

def build_agentic_indices():
    # 🧠 Configure LlamaIndex global settings to use Gemini Embeddings
    print("🧠 Configuring Gemini Embedding model...")
    Settings.embed_model = GeminiEmbedding(model_name="models/gemini-embedding-001")
    Settings.llm = None # We don't need the LLM just for generating embeddings

    # 1. Process n8n documentation
    print("📂 Processing n8n documentation...")
    if os.path.exists("docs/n8n") and os.listdir("docs/n8n"):
        n8n_docs = SimpleDirectoryReader("docs/n8n").load_data()
        n8n_index = VectorStoreIndex.from_documents(n8n_docs)
        # Persist storage to local disk
        n8n_index.storage_context.persist(persist_dir="storage/n8n")
        print("✅ n8n vector index created and saved to 'storage/n8n'")
    else:
        print("⚠️ 'docs/n8n' is empty or missing. Skipping n8n indexing.")

    # 2. Process Claude Code documentation
    print("📂 Processing Claude Code documentation...")
    if os.path.exists("docs/claude") and os.listdir("docs/claude"):
        claude_docs = SimpleDirectoryReader("docs/claude").load_data()
        claude_index = VectorStoreIndex.from_documents(claude_docs)
        # Persist storage to local disk
        claude_index.storage_context.persist(persist_dir="storage/claude")
        print("✅ Claude Code vector index created and saved to 'storage/claude'")
    else:
        print("⚠️ 'docs/claude' is empty or missing. Skipping Claude indexing.")

    print("🎉 Data ingestion pipeline for the agent completed successfully!")

if __name__ == "__main__":
    build_agentic_indices()
