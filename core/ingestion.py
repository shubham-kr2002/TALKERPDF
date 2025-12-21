import os
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from config.settings import CHROMA_DB_PATH, DATA_DOCS_PATH

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Initialize embedding model (small, free, runs locally)
embedding_model = SentenceTransformer('all-mpnet-base-v2')

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text

def sliding_window_chunker(text, chunk_size=500, overlap=100):
    """
    Split text into chunks using a sliding window approach.
    
    Args:
        text: The text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Number of overlapping characters between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk)
        
        # Move the window forward
        start += chunk_size - overlap
    
    return chunks

def get_existing_documents(collection):
    """Get list of already ingested document names from collection metadata."""
    try:
        # Get all items from collection
        all_items = collection.get()
        if all_items and all_items['metadatas']:
            # Extract unique source names
            existing_docs = set(meta['source'] for meta in all_items['metadatas'])
            return existing_docs
        return set()
    except Exception as e:
        print(f"Error checking existing documents: {e}")
        return set()

def ingest_pdfs(uploaded_files):
    """
    Process uploaded PDF files and store in ChromaDB.
    Prevents duplicate ingestion of already processed documents.
    """
    # Get or create collection
    collection = chroma_client.get_or_create_collection(
        name="rag_docs",
        metadata={"description": "RAG document chunks"}
    )
    
    # Check for existing documents
    existing_docs = get_existing_documents(collection)
    print(f"Already ingested documents: {existing_docs}")
    
    ingested_count = 0
    skipped_count = 0
    
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        
        # Skip if already ingested
        if filename in existing_docs:
            print(f"Skipping {filename} - already ingested")
            skipped_count += 1
            continue
        
        # Save uploaded file to data/docs
        pdf_path = os.path.join(DATA_DOCS_PATH, filename)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            print(f"Warning: No text extracted from {filename}")
            continue
        
        # Chunk text using sliding window
        chunks = sliding_window_chunker(text, chunk_size=500, overlap=100)
        print(f"Processing {filename}: {len(chunks)} chunks")
        
        # BATCH ENCODING: Process chunks in batches for 3-5x speedup
        # The CPU can process 32-64 chunks at once much faster than 1 by 1
        batch_size = 32
        
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            # Generate embeddings for entire batch at once (much faster!)
            batch_embeddings = embedding_model.encode(batch_chunks, show_progress_bar=False)
            
            # Prepare batch data for ChromaDB
            batch_ids = []
            batch_metadatas = []
            
            for i, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                chunk_idx = batch_start + i
                batch_ids.append(f"{filename}_chunk_{chunk_idx}")
                batch_metadatas.append({
                    "source": filename,
                    "chunk_id": chunk_idx,
                    "chunk_size": len(chunk)
                })
            
            # Add entire batch to ChromaDB at once
            collection.add(
                embeddings=batch_embeddings.tolist(),
                documents=batch_chunks,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        
        print(f"Successfully ingested {filename} with {len(chunks)} chunks")
        ingested_count += 1
    
    print(f"\nIngestion complete: {ingested_count} new documents, {skipped_count} skipped")
    return True
