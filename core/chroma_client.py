"""
Shared ChromaDB client module.
Uses EphemeralClient for Streamlit Cloud (no persistent disk storage).
Uses PersistentClient for local development.
"""
import os
import chromadb
from config.settings import CHROMA_DB_PATH

# Global client instance
_chroma_client = None

def is_streamlit_cloud():
    """Detect if running on Streamlit Cloud."""
    # Streamlit Cloud sets specific environment variables
    return (
        os.environ.get("STREAMLIT_SHARING") == "true" or
        os.path.exists("/mount/src") or  # Streamlit Cloud mount path
        os.environ.get("HOME", "").startswith("/home/adminuser")
    )

def get_chroma_client():
    """
    Get or create a ChromaDB client instance.
    
    For Streamlit Cloud: Uses EphemeralClient (in-memory) because:
    - Cloud has ephemeral storage that gets wiped between sessions
    - PersistentClient causes HNSW index corruption errors
    - Data is stored in session state instead
    
    For Local: Uses PersistentClient for data persistence across restarts.
    
    Returns:
        ChromaDB client instance
    """
    global _chroma_client
    
    if _chroma_client is not None:
        return _chroma_client
    
    if is_streamlit_cloud():
        print("🌐 Streamlit Cloud detected - using in-memory ChromaDB")
        # Use EphemeralClient for cloud - avoids disk persistence issues
        _chroma_client = chromadb.EphemeralClient()
    else:
        print("💻 Local environment detected - using persistent ChromaDB")
        # Use PersistentClient for local development
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    return _chroma_client


def reset_client():
    """Reset the client instance (useful for testing or cleanup)."""
    global _chroma_client
    _chroma_client = None


def get_or_create_collection(name="rag_docs", metadata=None):
    """
    Get or create a collection with proper error handling.
    
    Args:
        name: Collection name
        metadata: Optional collection metadata
    
    Returns:
        ChromaDB collection
    """
    client = get_chroma_client()
    
    if metadata is None:
        metadata = {"description": "RAG document chunks"}
    
    return client.get_or_create_collection(
        name=name,
        metadata=metadata
    )


def get_collection(name="rag_docs"):
    """
    Get an existing collection with error handling.
    
    Args:
        name: Collection name
    
    Returns:
        ChromaDB collection or None if not found
    """
    client = get_chroma_client()
    
    try:
        return client.get_collection(name=name)
    except Exception as e:
        print(f"Collection '{name}' not found: {e}")
        return None
