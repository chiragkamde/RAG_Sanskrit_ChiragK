import os
import sys
from pathlib import Path
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000")) # Changed to 8000 to match standard ChromaDB port
DATA_DIR = Path("./data")

def main():
    print("=" * 60)
    print("Digital Nalanda - Scripture Ingestion Pipeline")
    print("=" * 60)

    # 1. Ensure directory exists
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created directory at {DATA_DIR.absolute()}")
        print("Please drop your .txt or .md scripture files in there and run again.")
        sys.exit(0)

    # 2. Load Documents using LlamaIndex native reader
    print("Reading files from ./data...")
    documents = SimpleDirectoryReader(input_dir=str(DATA_DIR), required_exts=[".txt", ".md"]).load_data()
    
    if not documents:
        print("No .txt or .md documents found in the data directory.")
        sys.exit(0)
        
    print(f"Loaded {len(documents)} document pages/files.")

    # 3. Intelligent Scripture Chunking
    # chunk_size=512 tokens is roughly 380 words (perfect for a verse + commentary)
    # chunk_overlap=50 tokens ensures verses aren't cleanly severed from context
    text_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )

    # 4. Initialize ChromaDB
    print(f"\nConnecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")
    try:
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        chroma_collection = chroma_client.get_or_create_collection("digital_nalanda")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        print("Successfully connected to ChromaDB.")
    except Exception as e:
        print(f"CRITICAL: Could not connect to ChromaDB. Ensure your docker container is running. Error: {e}")
        sys.exit(1)

    # 5. Initialize Embedding Model (Routing to Host OS)
    print(f"Initializing Nomic Embeddings via host Ollama...")
    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url=OLLAMA_BASE_URL
    )

    # 6. Build the Index (This executes chunking, embedding, and storing)
    print("\nStarting ingestion process (this may take a moment depending on file size)...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[text_parser], # Apply our specific chunking rules
        show_progress=True
    )

    print("\n" + "=" * 60)
    print("Ingestion completely successful! Data is ready for querying.")
    print("=" * 60)

if __name__ == "__main__":
    main()