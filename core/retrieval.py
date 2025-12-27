import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from flashrank import Ranker, RerankRequest
from rank_bm25 import BM25Okapi
from core.chroma_client import get_chroma_client, get_collection

# Get shared ChromaDB client (handles Cloud vs Local automatically)
chroma_client = get_chroma_client()

# Global model instances (can be cached by UI)
_embedding_model = None
_ranker = None

# Global cache for BM25 to avoid rebuilding it every query
bm25_index = None
bm25_corpus = []  # List of text chunks
bm25_ids = []     # List of chunk IDs
bm25_metadatas = []  # List of metadata dicts

def get_embedding_model():
    """Get or create embedding model instance."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('all-mpnet-base-v2')
    return _embedding_model

def get_ranker():
    """Get or create FlashRank ranker instance."""
    global _ranker
    if _ranker is None:
        # FlashRank is ultra-lightweight and superfast (runs on CPU in milliseconds)
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp")
    return _ranker

def sigmoid(x):
    """Convert raw logits to probability (0-1) using sigmoid function."""
    return 1 / (1 + np.exp(-x))

def build_bm25_index(all_chunks, all_ids=None, all_metadatas=None):
    """
    Builds the keyword search index. Call this after ingestion!
    
    Args:
        all_chunks: List of text documents
        all_ids: List of document IDs (optional)
        all_metadatas: List of metadata dicts (optional)
    """
    global bm25_index, bm25_corpus, bm25_ids, bm25_metadatas
    
    if not all_chunks:
        print("No chunks provided for BM25 index.")
        return
    
    print(f"Building BM25 Keyword Index for {len(all_chunks)} chunks...")
    
    # Tokenize corpus (simple whitespace tokenization)
    tokenized_corpus = [doc.lower().split() for doc in all_chunks]
    bm25_index = BM25Okapi(tokenized_corpus)
    bm25_corpus = all_chunks
    bm25_ids = all_ids if all_ids else [str(i) for i in range(len(all_chunks))]
    bm25_metadatas = all_metadatas if all_metadatas else [{} for _ in all_chunks]
    
    print(f"✓ BM25 Index built successfully!")

def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    """
    Merges results from two sources based on their rank, not their raw score.
    This is standard in search engineering (RRF).
    
    Args:
        vector_results: List of document dicts from vector search
        bm25_results: List of document dicts from BM25 search
        k: Constant for RRF formula (default 60)
    
    Returns:
        Merged and sorted list of documents
    """
    global bm25_corpus, bm25_ids
    fusion_scores = {}
    
    # 1. Process Vector Results
    for rank, doc in enumerate(vector_results):
        doc_id = doc.get('id', doc.get('chunk_id', str(rank)))
        if doc_id not in fusion_scores:
            fusion_scores[doc_id] = {'doc': doc, 'score': 0}
        # The formula: 1 / (rank + k)
        fusion_scores[doc_id]['score'] += 1 / (rank + k)
    
    # 2. Process BM25 Results
    for rank, doc in enumerate(bm25_results):
        doc_id = doc.get('id', doc.get('chunk_id', str(rank)))
        
        if doc_id not in fusion_scores:
            fusion_scores[doc_id] = {'doc': doc, 'score': 0}
        
        fusion_scores[doc_id]['score'] += 1 / (rank + k)
    
    # 3. Sort by RRF Score (High is better)
    sorted_docs = sorted(fusion_scores.values(), key=lambda x: x['score'], reverse=True)
    
    # Return just the document objects with RRF score
    return [{'rrf_score': item['score'], **item['doc']} for item in sorted_docs]

def normalize_scores(results):
    """
    Normalize confidence scores to 0-100% range for better visualization.
    Uses Min-Max scaling to make top results look confident.
    """
    if not results:
        return []
    
    scores = [r.get('score', 0) for r in results]
    min_s, max_s = min(scores), max(scores)
    
    # Spread the scores out to fill 0% to 100%
    for res in results:
        if max_s == min_s:
            res['confidence'] = 1.0  # If all are same, trust them all
        else:
            # Mathematical normalization
            norm = (res['score'] - min_s) / (max_s - min_s)
            # Boost the curve slightly so the top result looks "Good" (>70%)
            res['confidence'] = norm  # Keep as 0-1 for internal use
    
    return results

def search_documents(query, k=15):
    """
    Perform vector search to get initial candidates from ChromaDB.
    
    Args:
        query: User's search query
        k: Number of results to retrieve (default 15 for re-ranking)
    
    Returns:
        Dictionary with documents, metadatas, and distances
    """
    try:
        # Get collection using shared client
        collection = get_collection(name="rag_docs")
        if collection is None:
            print("⚠️ Collection 'rag_docs' not found. Please upload documents first.")
            return None
        
        # Convert query to embedding
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.encode(query).tolist()
        
        # Query ChromaDB for top k results with retry logic
        results = None
        for attempt in range(3):
            try:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k
                )
                break
            except Exception as e:
                error_msg = str(e).lower()
                if "hnsw" in error_msg or "nothing found on disk" in error_msg:
                    if attempt < 2:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                raise
        
        return results
    
    except Exception as e:
        print(f"Error in vector search: {e}")
        return None

def rerank_results(query, results, top_n=5):
    """
    Re-rank search results using a cross-encoder for better accuracy.
    
    Args:
        query: User's search query
        results: Results from vector search
        top_n: Number of top results to return after re-ranking
    
    Returns:
        List of top_n highest-scoring documents
    """
    if not results or not results['documents'] or not results['documents'][0]:
        return []
    
    documents = results['documents'][0]
    metadatas = results['metadatas'][0] if results['metadatas'] else []
    
    # Use FlashRank for ultra-fast re-ranking (2s -> 0.2s)
    ranker = get_ranker()
    
    # Prepare passages for FlashRank
    passages = [{"id": idx, "text": doc} for idx, doc in enumerate(documents)]
    
    # Create rerank request
    rerank_request = RerankRequest(query=query, passages=passages)
    
    # Get ranked results (FlashRank is ~10x faster than CrossEncoder)
    ranked_results = ranker.rerank(rerank_request)
    
    # Combine documents with their scores and metadata
    scored_docs = []
    for result in ranked_results:
        idx = result['id']
        score = result['score']  # FlashRank returns normalized scores (0-1)
        doc = documents[idx]
        
        # Parse metadata and convert image_paths from JSON string to list
        metadata = metadatas[idx] if idx < len(metadatas) else {}
        if 'image_paths' in metadata and isinstance(metadata['image_paths'], str):
            try:
                metadata['image_paths'] = json.loads(metadata['image_paths'])
            except (json.JSONDecodeError, TypeError):
                metadata['image_paths'] = []
        
        scored_docs.append({
            'document': doc,
            'text': doc,  # Alias for compatibility
            'score': float(score),  # FlashRank score (already normalized)
            'confidence': float(score),  # Use directly as confidence (0-1)
            'metadata': metadata,
            'source': metadata.get('source', 'Unknown'),
            'chunk_id': metadata.get('chunk_id', 'Unknown')
        })
    
    # Already sorted by FlashRank, just take top_n
    top_results = scored_docs[:top_n]
    
    print(f"Re-ranking complete: Top {top_n} from {len(documents)} candidates")
    for i, result in enumerate(top_results, 1):
        print(f"  {i}. Confidence: {result['confidence']*100:.2f}% (Raw: {result['score']:.4f}) | Source: {result['metadata'].get('source', 'unknown')}")
    
    return top_results

def hybrid_search(query, k=10, max_retries=3):
    """
    Hybrid search combining Vector Search (semantic) and BM25 (keyword) with RRF.
    Optimized to only send top 10 candidates to re-ranker for maximum speed.
    
    Args:
        query: User's search query
        k: Number of candidates to retrieve from each method (default 10)
        max_retries: Number of retries for transient errors (default 3)
    
    Returns:
        List of reranked results with normalized confidence scores
    """
    global bm25_index, bm25_corpus, bm25_ids, bm25_metadatas
    
    try:
        # Use shared collection getter with proper error handling
        collection = get_collection(name="rag_docs")
        if collection is None:
            print("⚠️ Collection 'rag_docs' not found. Please upload documents first.")
            return []
        
        embedding_model = get_embedding_model()
        
        # 1. VECTOR SEARCH (Semantic) - Captures "Meaning"
        # ------------------------------------------------
        query_embedding = embedding_model.encode(query).tolist()
        
        # Retry logic for transient HNSW errors (common in cloud environments)
        vector_results = None
        last_error = None
        for attempt in range(max_retries):
            try:
                vector_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k
                )
                break  # Success, exit retry loop
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                # Check if it's a transient HNSW/disk error
                if "hnsw" in error_msg or "nothing found on disk" in error_msg:
                    print(f"⚠️ Vector search attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                        continue
                raise  # Re-raise non-transient errors immediately
        
        if vector_results is None:
            print(f"⚠️ Vector search failed after {max_retries} attempts: {last_error}")
            # Fall back to BM25-only search
            vector_results = {'documents': [[]], 'ids': [[]], 'metadatas': [[]]}
        
        # Convert to list of document dicts
        vector_docs = []
        if vector_results['documents'] and vector_results['documents'][0]:
            for i, doc in enumerate(vector_results['documents'][0]):
                vector_docs.append({
                    "text": doc,
                    "id": vector_results['ids'][0][i],
                    "metadata": vector_results['metadatas'][0][i] if vector_results['metadatas'] else {},
                })
        
        # 2. KEYWORD SEARCH (BM25) - Captures "Exact Words"
        # ------------------------------------------------
        bm25_docs = []
        if bm25_index is None:
            print("⚠️ BM25 Index not found. Using Vector only.")
        else:
            tokenized_query = query.lower().split()
            # Get scores for all documents
            bm25_scores = bm25_index.get_scores(tokenized_query)
            
            # Get top k indices
            top_k_indices = np.argsort(bm25_scores)[::-1][:k]
            
            for idx in top_k_indices:
                if bm25_scores[idx] > 0:  # Only include if there's a match
                    bm25_docs.append({
                        "text": bm25_corpus[idx],
                        "id": bm25_ids[idx],
                        "metadata": bm25_metadatas[idx],
                    })
        
        # 3. RECIPROCAL RANK FUSION (RRF) - Merge intelligently
        # -----------------------------------------------------
        merged_results = reciprocal_rank_fusion(vector_docs, bm25_docs, k=60)
        
        # 4. OPTIMIZATION: Only send top 10 to re-ranker
        # ----------------------------------------------
        # This ensures latency stays constant regardless of dataset size
        top_candidates = merged_results[:10]
        
        if not top_candidates:
            return []
        
        print(f"RRF Fusion: {len(merged_results)} unique docs, sending top 10 to re-ranker")
        
        # 5. RERANK with FlashRank (The slow part, now optimized)
        # -------------------------------------------------------
        ranker = get_ranker()
        passages = [{"id": i, "text": doc['text']} for i, doc in enumerate(top_candidates)]
        rerank_request = RerankRequest(query=query, passages=passages)
        ranked_results = ranker.rerank(rerank_request)
        
        # 6. Map back to full document info
        # ---------------------------------
        final_results = []
        for result in ranked_results[:5]:  # Top 5
            idx = result['id']
            doc = top_candidates[idx]
            final_results.append({
                'document': doc['text'],
                'text': doc['text'],
                'score': float(result['score']),
                'confidence': float(result['score']),
                'metadata': doc.get('metadata', {}),
                'source': doc.get('metadata', {}).get('source', 'Unknown'),
                'chunk_id': doc.get('metadata', {}).get('chunk_id', 'Unknown')
            })
        
        # 7. Normalize confidence scores for better visualization
        # -------------------------------------------------------
        final_results = normalize_scores(final_results)
        
        print(f"Hybrid Search complete: Top 5 results")
        for i, result in enumerate(final_results, 1):
            print(f"  {i}. Confidence: {result['confidence']*100:.1f}% | Source: {result['source']}")
        
        return final_results
        
    except Exception as e:
        print(f"Error in hybrid search: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to regular vector search
        try:
            search_results = search_documents(query, k=10)
            if search_results:
                return rerank_results(query, search_results, top_n=5)
        except:
            pass
        return []

def retrieve_context(query, initial_k=15, top_k=5):
    """
    Full retrieval pipeline: Hybrid Search (Vector + BM25) + Re-ranking.
    
    Args:
        query: User's search query
        initial_k: Number of candidates to retrieve from each search method
        top_k: Number of final results after re-ranking
    
    Returns:
        Concatenated context string from top results
    """
    try:
        # Use hybrid search for best results
        reranked_results = hybrid_search(query, k=initial_k)
        
        if not reranked_results:
            return "No relevant context found."
        
        # Extract documents and concatenate
        contexts = [result['document'] for result in reranked_results[:top_k]]
        return "\n\n".join(contexts)
    
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return "Error retrieving context from database."
