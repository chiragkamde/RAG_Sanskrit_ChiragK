import streamlit as st
import os
import json
import requests
from typing import List, Dict, Any
from simple_rag_demo import VedicRAGDemo

st.set_page_config(
    page_title="Sanskrit Document Retrieval-Augmented Generation (RAG) System",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .verse-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_rag_system():
    """Load the RAG system once (FAISS + flan-t5, CPU). Works with corpus or upload-only."""
    corpus_dirs = ("nalanda_library", "vedic_texts")
    parent = os.path.dirname(os.getcwd())
    for d in corpus_dirs:
        for path in (d, os.path.join(parent, d)):
            if os.path.exists(path):
                return VedicRAGDemo(corpus_dir=os.path.abspath(path), top_k=3)
    return VedicRAGDemo(corpus_dir=None, top_k=3)

def check_ollama_connection(base_url: str) -> bool:
    """Check if Ollama server is running"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

def get_ollama_models(base_url: str) -> List[str]:
    """Get list of available Ollama models"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
    except Exception:
        pass
    return []

def query_ollama(model: str, prompt: str, base_url: str) -> str:
    """Query Ollama model with streaming response"""
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7,
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json().get('response', 'No response generated')
        return f"Error: {response.status_code}"
    except requests.exceptions.Timeout:
        return "Error: Request timeout. Ollama server may be slow or unresponsive."
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.title("Sanskrit Document Retrieval-Augmented Generation (RAG) System")
    st.markdown("*Explore ancient Vedic wisdom — Sanskrit or transliterated queries*")

    # File upload: PDF, TXT, DOCX (Sanskrit or English)
    st.subheader("📄 Upload documents (optional)")
    st.caption("Upload PDF, TXT, or DOCX (Sanskrit or English). Questions will be answered from your uploads.")
    uploaded = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="PDF, TXT, DOCX supported. After uploading, click 'Use uploaded documents' then ask questions.",
    )
    use_uploads = st.checkbox("Use uploaded documents for Q&A", value=False, help="When checked, answers are based only on your uploaded files.")

    rag_system = load_rag_system()
    if use_uploads and uploaded:
        upload_key = tuple((f.name, f.size) for f in uploaded)
        if st.session_state.get("upload_key") != upload_key:
            try:
                rag_system.build_from_uploaded_files(uploaded)
                st.session_state["upload_key"] = upload_key
                st.success(f"✅ Loaded {len(uploaded)} file(s). Ask questions below based on your documents.")
            except Exception as e:
                st.error(f"Failed to process uploads: {e}")
        else:
            st.success(f"✅ Using {len(uploaded)} uploaded file(s). Ask questions below.")
    elif use_uploads and not uploaded:
        st.warning("Upload at least one PDF, TXT, or DOCX file and try again.")
    else:
        if "upload_key" in st.session_state:
            del st.session_state["upload_key"]
        if rag_system.has_uploaded_documents():
            rag_system.clear_uploaded_documents()

    with st.sidebar:
        st.header("⚙️ Configuration")

        use_ollama = st.checkbox(
            "Use Ollama for generation (optional)",
            value=False,
            help="If unchecked, uses built-in flan-t5 (CPU, no Ollama required)."
        )

        ollama_url = st.text_input(
            "Ollama Server URL",
            value="http://localhost:11434",
            help="Only used when 'Use Ollama' is checked."
        )

        is_connected = check_ollama_connection(ollama_url) if use_ollama else False
        if use_ollama:
            if is_connected:
                st.success("✅ Connected to Ollama")
            else:
                st.error("❌ Cannot connect to Ollama. Uncheck to use built-in flan-t5.")

        available_models = get_ollama_models(ollama_url) if is_connected else []
        if use_ollama and available_models:
            selected_model = st.selectbox("Select Model", available_models)
        else:
            selected_model = None

        st.divider()
        st.subheader("RAG Settings")
        max_verses = st.slider(
            "Max chunks to retrieve",
            min_value=1,
            max_value=10,
            value=3,
            help="Top-k retrieved passages (pipeline uses k=3 internally)."
        )
        use_rag = st.checkbox("Use RAG context", value=True)
        st.divider()
        st.subheader("About")
        st.markdown("""
        - **Built-in (default)**: FAISS + flan-t5, CPU-only, .txt/.pdf ingestion
        - **Ollama**: Optional; use your local Ollama model for generation
        - Queries in Sanskrit or transliterated Sanskrit supported
        - **First search** may take 5–15 min if the index is building; later searches are fast.
        """)

    rag_system = load_rag_system()

    # Require Ollama only if user explicitly chose it
    if use_ollama and (not is_connected or not selected_model):
        st.warning("⚠️ Turn off 'Use Ollama' to run with built-in flan-t5, or start Ollama and select a model.")
        return

    if not use_uploads and not rag_system.corpus_dir:
        st.info("👆 **Upload PDF/TXT/DOCX** above and check 'Use uploaded documents', or run VedicDatasetGenerator.py to create nalanda_library/.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        user_query = st.text_input(
            "💭 Ask in Sanskrit or English (transliterated OK):",
            placeholder="e.g., धर्म किम्? or What is duty?",
            help="Sanskrit or transliterated Sanskrit supported."
        )
    with col2:
        submit_button = st.button("🔍 Search", use_container_width=True)

    if submit_button and user_query:
        with st.spinner("🔄 Processing..."):
            st.divider()

            if use_ollama and selected_model:
                # Retrieval only, then Ollama for generation
                sources, retrieval_time = rag_system.retrieve_only(user_query)
                context = "\n\n".join([s.get("text", "") for s in sources]) if use_rag else ""
                prompt = f"""Context:\n{context}\n\nQuestion:\n{user_query}\n\nAnswer in Sanskrit or English explanation."""
                import time as _time
                gen_start = _time.perf_counter()
                response = query_ollama(selected_model, prompt, ollama_url)
                generation_time = _time.perf_counter() - gen_start
                total_time = retrieval_time + generation_time
                result = {"answer": response, "sources": sources, "retrieval_time": retrieval_time, "generation_time": generation_time, "total_time": total_time}
            else:
                # Full pipeline: retrieval + flan-t5 (CPU)
                result = rag_system.query(user_query)
                response = result.get("answer", "No answer generated.")
                retrieval_time = result.get("retrieval_time", 0)
                generation_time = result.get("generation_time", 0)
                total_time = result.get("total_time", 0)

            sources = result.get("sources", [])

            st.subheader("📚 Retrieved context (top-k)")
            if sources:
                for i, src in enumerate(sources[:max_verses], 1):
                    with st.container():
                        meta = src.get("metadata", {})
                        st.markdown(f"**{i}. {meta.get('source', 'Chunk')}**")
                        st.text(src.get("text", "")[:500] + ("..." if len(src.get("text", "")) > 500 else ""))
                        st.divider()
            else:
                st.caption("No chunks retrieved.")

            st.subheader("🤖 Answer")
            st.markdown(response)

            st.caption(
                f"⏱️ Retrieval: {retrieval_time:.2f}s | Generation: {generation_time:.2f}s | Total: {total_time:.2f}s"
            )
            st.divider()

            if st.button("💾 Save conversation"):
                conversation = {
                    "query": user_query,
                    "response": response,
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                    "total_time": total_time,
                    "sources_count": len(sources),
                }
                log_file = "conversations.jsonl"
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(conversation, ensure_ascii=False) + "\n")
                st.success("✅ Saved to " + log_file)

if __name__ == "__main__":
    main()
