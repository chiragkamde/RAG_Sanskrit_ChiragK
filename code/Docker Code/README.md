# Digital Nalanda - Local RAG API for Ancient Indian Scriptures

A complete, decoupled Proof of Concept (POC) for a local Retrieval-Augmented Generation (RAG) API backend designed to ingest, chunk, and query ancient Indian scriptures (Vedas, Upanishads, etc.).

## Architecture Overview

The system follows a classic RAG pipeline:

1. **Data Ingestion**: Text/Markdown files from `./data` directory
2. **Chunking**: Intelligent scripture-aware chunking (preserves verses/shlokas)
3. **Embedding**: `nomic-embed-text` model via Ollama
4. **Vector Storage**: ChromaDB for persistent vector database
5. **Retrieval**: Semantic similarity search
6. **Generation**: `gtp-oss:20B` model via Ollama for answer generation

## Prerequisites

- **Ollama** running natively on host machine (port 11434)
  - Model: `gtp-oss:20B` (for generation)
  - Model: `nomic-embed-text` (for embeddings)
- **Docker & Docker Compose** installed
- **Python 3.11+** (for running ingest.py locally, optional)

## Project Structure

```
.
├── requirements.txt          # Python dependencies
├── Dockerfile               # FastAPI container definition
├── docker-compose.yml       # Multi-container orchestration
├── main.py                  # FastAPI server with /query endpoint
├── ingest.py               # Data ingestion & chunking script
├── README.md               # This file
└── data/                   # Scripture data directory (create manually)
    ├── vedas.txt
    ├── upanishads.md
    └── ...
```

## Setup Instructions

### 1. Prepare Data Directory

Create a `data` folder in the project root and add your scripture files:

```bash
mkdir data
# Add your .txt or .md files to the data/ directory
```

**Supported formats**: `.txt`, `.md`

**Example structure**:
```
data/
├── rigveda.txt
├── yajurveda.txt
├── samaveda.txt
├── atharvaveda.txt
├── upanishads.md
└── brahmasutras.txt
```

### 2. Start Docker Services

From the project root directory:

```bash
docker-compose up -d
```

This will:
- Start ChromaDB service on port 8001 (internal: 8000)
- Build and start FastAPI container on port 8000
- Create persistent volume for ChromaDB data

Verify services are running:
```bash
docker-compose ps
```

### 3. Ingest Scripture Data

**Option A: Run inside Docker container**

```bash
docker-compose exec fastapi python ingest.py
```

**Option B: Run locally (requires Python 3.11+ and dependencies)**

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (if running locally)
export OLLAMA_BASE_URL=http://localhost:11434
export CHROMA_HOST=localhost
export CHROMA_PORT=8001

# Run ingestion
python ingest.py
```

The ingestion script will:
- Load all `.txt` and `.md` files from `./data`
- Chunk them intelligently (preserving logical boundaries like verses)
- Extract metadata (filename, chunk index)
- Embed chunks using `nomic-embed-text`
- Store in ChromaDB collection named "nalanda"

### 4. Query the API

Once ingestion is complete, the API is ready to accept queries.

**Health Check**:
```bash
curl http://localhost:8000/health
```

**Query Endpoint**:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the four Vedas?"}'
```

**Response Format**:
```json
{
  "answer": "The four Vedas are...",
  "sources": [
    {
      "text": "Retrieved scripture excerpt...",
      "metadata": {
        "source_file": "vedas.txt",
        "source_path": "vedas.txt",
        "chunk_index": 0,
        "total_chunks": 45
      }
    }
  ]
}
```

## API Endpoints

### `GET /`
Root endpoint with service information.

### `GET /health`
Health check endpoint. Returns service status and configuration.

### `POST /query`
Main RAG query endpoint.

**Request Body**:
```json
{
  "question": "Your question about the scriptures"
}
```

**Response**:
```json
{
  "answer": "Generated answer from LLM",
  "sources": [
    {
      "text": "Retrieved context from scriptures",
      "metadata": {...}
    }
  ]
}
```

## Configuration

### Environment Variables

Set these in `docker-compose.yml` or `.env`:

- `OLLAMA_BASE_URL`: Ollama service URL (default: `http://host.docker.internal:11434`)
- `CHROMA_HOST`: ChromaDB hostname (default: `chroma`)
- `CHROMA_PORT`: ChromaDB port (default: `8000`)

### Chunking Strategy

The `ingest.py` script uses intelligent chunking:

1. **Primary split**: By double newlines (paragraphs/verses)
2. **Secondary split**: By sentence boundaries (". ")
3. **Fallback**: By character limit (512 chars default)

This preserves Shlokas, Sutras, and other logical units.

**Adjust chunking parameters in `ingest.py`**:
```python
CHUNK_SIZE = 512      # Target chunk size in characters
CHUNK_OVERLAP = 50    # Overlap between chunks (not used in current implementation)
```

## Troubleshooting

### ChromaDB Connection Error
```
Error connecting to ChromaDB: Connection refused
```
**Solution**: Ensure ChromaDB is running:
```bash
docker-compose ps
docker-compose logs chroma
```

### Ollama Connection Error
```
Error: Cannot connect to Ollama at http://host.docker.internal:11434
```
**Solution**: 
- Verify Ollama is running on host machine
- Check Ollama is accessible on port 11434
- On Windows/Mac, `host.docker.internal` should resolve correctly
- On Linux, may need to use `--network host` in docker-compose

### No Data Ingested
```
No documents found to ingest.
```
**Solution**: 
- Verify `data/` directory exists and contains `.txt` or `.md` files
- Check file permissions
- Ensure files are UTF-8 encoded

### Slow Query Response
- First query may be slow (model loading)
- Subsequent queries should be faster
- Adjust `similarity_top_k` in `main.py` to retrieve fewer context chunks

## Development & Customization

### Modify LLM Parameters

Edit `main.py` in the `lifespan` function:
```python
llm = OllamaLLM(
    model="gtp-oss:20B",
    base_url=OLLAMA_BASE_URL,
    temperature=0.7,        # Adjust creativity (0.0-1.0)
    context_window=4096     # Adjust context size
)
```

### Change Retrieval Strategy

Edit `main.py` in the `/query` endpoint:
```python
query_engine = index.as_query_engine(
    similarity_top_k=3,           # Number of chunks to retrieve
    response_mode="compact"       # or "tree_summarize", "refine"
)
```

### Add Custom Metadata

Modify `create_documents_with_metadata()` in `ingest.py` to extract and attach custom metadata from files.

## Performance Notes

- **First ingestion**: May take several minutes depending on data size
- **First query**: May take 30-60 seconds (model loading)
- **Subsequent queries**: 5-15 seconds depending on context size
- **Vector storage**: Persistent across container restarts

## Stopping Services

```bash
docker-compose down
```

To also remove volumes (deletes ChromaDB data):
```bash
docker-compose down -v
```

## Next Steps

1. Add your scripture data to `./data`
2. Run ingestion: `docker-compose exec fastapi python ingest.py`
3. Query the API via `/query` endpoint
4. Integrate with frontend applications via the REST API

## License

This is a Proof of Concept for educational purposes.
