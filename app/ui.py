import os
import sys
import json
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

# Helper function to parse image_paths from metadata
def parse_image_paths(metadata):
    """Parse image_paths from metadata, handling both JSON strings and lists."""
    image_paths = metadata.get('image_paths', [])
    if isinstance(image_paths, str):
        try:
            return json.loads(image_paths)
        except (json.JSONDecodeError, TypeError):
            return []
    return image_paths if isinstance(image_paths, list) else []

# -----------------------------
# Design / CSS
# -----------------------------
_CUSTOM_CSS = r"""
:root{
    --bg:#0f1724; /* deep navy */
    --card:#111827; /* slightly lighter */
    --muted:#9ca3af;
    --accent:#06b6d4;
    --accent-2:#3b82f6;
    --radius:12px;
}
html, body, [class*="css"]  {
    background: linear-gradient(180deg, rgba(17,24,39,0.75), rgba(7,10,25,0.85));
    color: #e6eef8;
}
.talker-header{ display:flex; align-items:center; justify-content:center; gap:12px; padding:18px; border-radius:var(--radius); margin-bottom:18px}
.brand-title{ font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; font-weight:700; font-size:20px}
.brand-sub{ color:var(--muted); font-size:12px}
.card{ background: rgba(255,255,255,0.03); padding:16px; border-radius:var(--radius); box-shadow: 0 6px 18px rgba(2,6,23,0.6); border: 1px solid rgba(255,255,255,0.03);}
.upload-zone{ border:2px dashed rgba(255,255,255,0.06); padding:18px; text-align:center; border-radius:10px; transition: all .22s ease;}
.upload-zone:hover{ transform: translateY(-3px); box-shadow:0 8px 20px rgba(2,6,23,0.6); border-color: rgba(59,130,246,0.6)}
.btn-primary{ background: linear-gradient(90deg,var(--accent-2),var(--accent)); color:white; padding:8px 14px; border-radius:10px; border:none}
.chat-area{ display:flex; flex-direction:column; gap:12px;}
.bubble{ max-width:82%; padding:12px 16px; border-radius:18px; line-height:1.35; transition: transform .18s ease;}
.bubble.user{ background: linear-gradient(90deg, rgba(59,130,246,0.12), rgba(6,182,212,0.08)); color: #dbeafe; margin-left:auto; border-bottom-right-radius:6px}
.bubble.assistant{ background: rgba(255,255,255,0.02); color:#e6eef8; margin-right:auto; border-bottom-left-radius:6px}
.meta{ color:var(--muted); font-size:12px; margin-top:6px}
.small{ font-size:13px }
.floating-brand{ text-align:center; padding:8px; opacity:0.9 }
/* Accessibility focus */
input:focus, textarea:focus, button:focus { outline: 3px solid rgba(6,182,212,0.18); outline-offset:2px }
@media (max-width: 800px){ .bubble{ max-width:96% }}
"""

def inject_css():
        st.markdown(f"<style>{_CUSTOM_CSS}</style>", unsafe_allow_html=True)

inject_css()  # Call the function to inject CSS
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
    from core.chroma_client import get_chroma_client
    
    try:
        client = get_chroma_client()
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
    # Elegant centered header with subtle accent
    st.markdown(
        """
        <div class="talker-header card">
            <div style="display:flex;flex-direction:column;align-items:center;">
                <div class="brand-title">TALKER PDF</div>
                <div class="brand-sub">RAG • Groq• ChromaDB — Chat with PDFs</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
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
                        
                        # Rebuild BM25 index after ingestion
                        st.info("🔧 Rebuilding search index...")
                        initialize_search_engine.clear()  # Clear cache
                        initialize_search_engine()  # Rebuild index
                        
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
    
    # Display chat messages (bubble style) inside a visual card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        meta = message.get("debug_info", None)
        # Build bubble HTML
        bubble_class = 'assistant' if role != 'user' else 'user'
        avatar = '🤖' if role != 'user' else '👤'
        html = f"""
        <div class='chat-area'>
          <div class='bubble {bubble_class}'>
            <div style='display:flex;gap:10px;align-items:flex-start;'>
              <div style='font-size:18px'>{avatar}</div>
              <div class='small'>{content}</div>
            </div>
          </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
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
                
                # Check if user is asking for visual content
                show_images_inline = any(keyword in prompt.lower() for keyword in 
                    ['show', 'display', 'diagram', 'chart', 'graph', 'image', 'picture', 'figure', 'visual'])
                
                # Display images inline if user asks for them
                if show_images_inline:
                    for result in reranked_results[:3]:
                        image_paths = parse_image_paths(result['metadata'])
                        if image_paths:
                            st.markdown("---")
                            st.markdown("📊 **Relevant Charts/Diagrams:**")
                            for img_path in image_paths:
                                if os.path.exists(img_path):
                                    st.image(
                                        img_path,
                                        caption=f"Source: {result['metadata'].get('source', 'unknown')} - Page {result['metadata'].get('page', '?')}",
                                        use_container_width=True
                                    )
                            break  # Only show images from the top result
                
                # Debug view
                with st.expander("🔍 Debug Retrieval & Sources"):
                    st.caption(f"Top {min(3, len(reranked_results))} chunks used for generation:")
                    for i, result in enumerate(reranked_results[:3], 1):
                        st.markdown(f"**Chunk {i}** (Score: {result['score']:.4f})")
                        st.markdown(f"*Source: {result['metadata'].get('source', 'unknown')} - Page {result['metadata'].get('page', '?')}*")
                        
                        # Display images if available
                        image_paths = parse_image_paths(result['metadata'])
                        if image_paths:
                            st.markdown("📊 **Charts/Diagrams from this page:**")
                            for img_path in image_paths:
                                if os.path.exists(img_path):
                                    st.image(
                                        img_path,
                                        caption=f"Source: {result['metadata'].get('source', 'unknown')} - Page {result['metadata'].get('page', '?')}",
                                        use_container_width=True
                                    )
                        
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
