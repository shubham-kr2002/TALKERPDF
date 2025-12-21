import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.ingestion import ingest_pdfs
from core.retrieval import retrieve_context, search_documents, rerank_results, get_embedding_model, get_ranker, build_bm25_index, hybrid_search
from core.generation import generate_answer, contextualize_query

# Load environment variables
load_dotenv()

# Cache models to prevent reloading on every interaction
@st.cache_resource
def load_models():
    """Load and cache embedding model and ranker to prevent 5-second reload."""
    embedding_model = get_embedding_model()
    ranker = get_ranker()
    return embedding_model, ranker

@st.cache_resource
def initialize_search_engine():
    """Initialize BM25 keyword search index from existing ChromaDB data."""
    import chromadb
    client = chromadb.PersistentClient(path="data/chroma_db")
    
    try:
        collection = client.get_or_create_collection("rag_docs")
        existing_data = collection.get()  # Fetches everything
        
        if existing_data['documents']:
            print(f"Initializing search with {len(existing_data['documents'])} documents...")
            build_bm25_index(
                existing_data['documents'],
                existing_data['ids'],
                existing_data['metadatas']
            )
        else:
            print("Database is empty. Please ingest documents.")
    except Exception as e:
        print(f"Error initializing search engine: {e}")

# Initialize cached models
embedding_model, ranker = load_models()

# Initialize search engine once
initialize_search_engine()

# Page configuration
st.set_page_config(
    page_title="TALKER PDF",
    page_icon="📄",
    layout="wide"
)

def main():
    st.title("📄 TALKER PDF")
    st.markdown("*RAG Application with OpenRouter & ChromaDB*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Input
        api_key = st.text_input(
            "OpenRouter API Key",
            value=os.getenv("OPENROUTER_API_KEY", ""),
            type="password",
            help="Enter your OpenRouter API key"
        )
        
        # Update environment variable if provided
        if api_key:
            os.environ["OPENROUTER_API_KEY"] = api_key
        
        st.divider()
        
        # PDF Upload Section
        st.header("📁 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF documents"
        )
        
        if st.button("🔄 Ingest PDFs", use_container_width=True):
            if not uploaded_files:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Ingesting PDFs..."):
                    try:
                        ingest_pdfs(uploaded_files)
                        st.success(f"✅ Successfully ingested {len(uploaded_files)} PDF(s)!")
                    except Exception as e:
                        st.error(f"❌ Error ingesting PDFs: {str(e)}")
        
        st.divider()
        st.caption("Made with ❤️ using Streamlit")
    
    # Main Chat Area
    st.header("💬 Chat with your Documents")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize retrieval context storage
    if "last_retrieval" not in st.session_state:
        st.session_state.last_retrieval = None
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show debug info if available
            if message["role"] == "assistant" and "debug_info" in message:
                with st.expander("🔍 Debug Retrieval"):
                    debug_info = message["debug_info"]
                    st.caption(f"Retrieved {len(debug_info)} chunks")
                    for i, chunk_info in enumerate(debug_info[:3], 1):
                        st.markdown(f"**Chunk {i}** (Score: {chunk_info['score']:.4f})")
                        st.markdown(f"*Source: {chunk_info['metadata'].get('source', 'unknown')}*")
                        st.text_area(
                            f"Content {i}",
                            chunk_info['document'],
                            height=100,
                            key=f"debug_{message.get('timestamp', i)}_{i}",
                            disabled=True
                        )
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Check if API key is set
        if not os.getenv("OPENROUTER_API_KEY"):
            st.error("⚠️ Please set your OpenRouter API Key in the sidebar.")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            try:
                # STEP A: Contextualize (Rewrite) the Query
                with st.spinner("🔎 Refining query..."):
                    # Pass history excluding the just-added prompt
                    search_query = contextualize_query(prompt, st.session_state.messages[:-1])
                    # Show the rewrite for debugging
                    st.write(f"Searching for: '{search_query}'")

                # STEP B: Retrieve using the REWRITTEN query
                with st.spinner("🔍 Retrieving documents..."):
                    reranked_results = hybrid_search(search_query)
                    if not reranked_results:
                        st.error("No documents found. Please ingest PDFs first.")
                        return
                    st.session_state.last_retrieval = reranked_results

                # STEP C: Generate Answer (use original prompt for generation)
                with st.spinner("💭 Synthesizing answer..."):
                    context_chunks = reranked_results
                    response = generate_answer(prompt, context_chunks)
                    response_placeholder.markdown(response)
                
                # Debug view
                with st.expander("🔍 Debug Retrieval"):
                    st.caption(f"Top {min(3, len(reranked_results))} chunks used for generation:")
                    for i, result in enumerate(reranked_results[:3], 1):
                        st.markdown(f"**Chunk {i}** (Score: {result['score']:.4f})")
                        st.markdown(f"*Source: {result['metadata'].get('source', 'unknown')}*")
                        st.text_area(
                            f"Content {i}",
                            result['document'],
                            height=100,
                            key=f"current_debug_{i}",
                            disabled=True
                        )
                        st.divider()
                
                # Add assistant response to chat history with debug info
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "debug_info": reranked_results,
                    "timestamp": len(st.session_state.messages)
                })
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()
