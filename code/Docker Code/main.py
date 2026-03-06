import os
from contextlib import asynccontextmanager
from typing import List, Dict, Any
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama.base import Ollama as OllamaLLM
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.storage import StorageContext

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000")) # FIXED: Matched to standard 8000 port

chroma_client = None
index = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chroma_client, index
    
    print(f"Initializing ChromaDB connection at {CHROMA_HOST}:{CHROMA_PORT}...")
    try:
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        print("Connected to ChromaDB")
    except Exception as e:
        print(f"Warning: Could not connect to ChromaDB: {e}")
        print("The API will attempt to connect on first query")
    
    print(f"Initializing Ollama LLM from {OLLAMA_BASE_URL}...")
    llm = OllamaLLM(
        model="gpt-oss:20b", # FIXED: Corrected spelling
        base_url=OLLAMA_BASE_URL,
        request_timeout=300.0,
        temperature=0.1, # LOWERED: 0.1 prevents the AI from "hallucinating" fake scripture
        context_window=4096,
        system_prompt=(
            "You are a wise, highly accurate scholar of Ancient Indian Scriptures. "
            "You are the core intelligence of 'Digital Nalanda'. "
            "Answer the user's question ONLY using the provided retrieved context. "
            "If the answer is not contained in the context, state clearly that the "
            "information is not in the current Digital Nalanda archives. Do not guess."
        )
    )
    
    print(f"Initializing Ollama embeddings from {OLLAMA_BASE_URL}...")
    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url=OLLAMA_BASE_URL
    )
    
    # Set LlamaIndex globals
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    yield
    
    print("Shutting down Digital Nalanda API")


app = FastAPI(
    title="Digital Nalanda",
    description="Local RAG API for Ancient Indian Scriptures",
    version="1.0.0",
    lifespan=lifespan
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

def get_index():
    """Get or create the VectorStoreIndex."""
    global chroma_client, index
    
    if index is not None:
        return index
        
    if chroma_client is None:
        try:
            chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Cannot connect to ChromaDB: {e}")
            
    try:
        # FIXED: Collection name now exactly matches ingest.py
        vector_store = ChromaVectorStore(
            chroma_collection=chroma_client.get_or_create_collection("digital_nalanda")
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
        return index
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing index: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Digital Nalanda RAG API",
        "ollama_base_url": OLLAMA_BASE_URL,
        "chroma_host": CHROMA_HOST,
        "chroma_port": CHROMA_PORT
    }

@app.post("/query", response_model=QueryResponse)
async def query_scriptures(request: QueryRequest) -> QueryResponse:
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
        
    try:
        current_index = get_index()
        
        # Configure the query engine
        query_engine = current_index.as_query_engine(
            similarity_top_k=3,
            response_mode="compact"
        )
        
        response = query_engine.query(request.question)
        
        # Package the source metadata for the frontend
        sources = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node in response.source_nodes:
                source_info = {
                    "text": node.get_content()[:200] + "...", # Truncate for cleaner JSON output
                    "metadata": node.metadata if hasattr(node, 'metadata') else {}
                }
                sources.append(source_info)
                
        return QueryResponse(
            answer=str(response),
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/")
async def root():
    return {
        "service": "Digital Nalanda",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)