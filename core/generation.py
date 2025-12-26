import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Get API key from Streamlit secrets or environment variable
def get_api_key():
    """Get API key from Streamlit secrets (cloud) or environment (local)."""
    try:
        import streamlit as st
        # Try to get from Streamlit secrets first
        if hasattr(st, 'secrets') and "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except (ImportError, FileNotFoundError, AttributeError, RuntimeError):
        pass
    # Fall back to environment variable
    return os.getenv("GROQ_API_KEY")

# Lazy initialization of Groq client
_client = None

def get_client():
    """Lazy initialization of Groq client to avoid import-time crashes."""
    global _client
    if _client is None:
        api_key = get_api_key()
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found! Please set it in Streamlit secrets or .env file.\n"
                "For Streamlit Cloud: Add GROQ_API_KEY in App Settings > Secrets\n"
                "For local: Add GROQ_API_KEY=your_key to .env file"
            )
        _client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        )
    return _client

def contextualize_query(query, chat_history):
    """
    Uses the LLM to rewrite a query like "it" or "that" into a full question
    based on the previous chat history.
    """
    # If no history, no need to rewrite
    if not chat_history:
        return query

    # Construct a string of the last 2-3 turns to save tokens
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-4:]])
    
    system_prompt = """
    You are a Query Refiner. Your ONLY job is to rewrite the User's last question 
    so it can be understood without the chat history.
    
    RULES:
    1. Replace pronouns (it, he, she, they) with specific names/nouns from history.
    2. Keep the meaning exactly the same.
    3. Output ONLY the rewritten question. No "Here is the rewritten question:." 
    """
    
    user_prompt = f"""
    Chat History:
    {history_text}
    
    Latest User Question: 
    {query}
    
    Standalone Question:
    """

    try:
        response = get_client().chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return query

def generate_answer(query, context_chunks):
    """
    Generate an answer using Groq API with Llama 4 Scout model.
    Includes confidence-based hallucination prevention.
    
    Args:
        query: User's question
        context_chunks: Retrieved context (string or list of dicts with 'text' and 'confidence' keys)
    
    Returns:
        Generated answer string
    """
    # FILTER LOGIC: Check confidence before generating
    # If the top chunk's confidence is less than 25% (0.25), abort to prevent hallucination
    if isinstance(context_chunks, list) and len(context_chunks) > 0:
        # Check if the best result has sufficient confidence
        top_confidence = context_chunks[0].get('confidence', 1.0)
        if top_confidence < 0.25:
            return "⚠️ Low Confidence: I couldn't find any relevant information in the documents to answer your question."
    elif not context_chunks:
        return "⚠️ Low Confidence: I couldn't find any relevant information in the documents to answer your question."
    
    # Format the context for the LLM
    # Handle both string context and list of chunk dictionaries
    if isinstance(context_chunks, str):
        context_text = context_chunks
    elif isinstance(context_chunks, list):
        context_text = "\n\n".join([
            f"Source (Chunk {c.get('chunk_id', c.get('metadata', {}).get('chunk_id', 'unknown'))}): {c.get('text', c.get('document', str(c)))}" 
            for c in context_chunks
        ])
    else:
        context_text = str(context_chunks)
    
    system_prompt = """You are an intelligent Research Assistant. 
    Your goal is to answer clearly and then **guide the user deeper** and **spark curiosity** into the specific topics you just mentioned.

    STRUCTURE YOUR RESPONSE LIKE THIS:
    
    ### RESPONSE FORMAT:
    
    1. **The Answer:** - Provide a clear, direct answer to the user's question using the Context.
       - Use specific terms, dates, and names from the documents.
       - Cite sources like [1].
    
    2. **suggested_questions:**
       - At the very bottom, add a section titled "**Questions to Ask Next:**"
       - Look at the specific concepts, algorithms, or entities you mentioned in your answer.
       - Create 1 short, bold follow-up questions that drill down into those specifics.
       
    ### EXAMPLE BEHAVIOR:
    - If you just explained "Machine Learning", your next questions should be: **"What is Linear Regression?"** or **"How does a Neural Network work?"**
    - If you just explained "Revenue Growth", your next questions should be: **"What drove the Q3 sales spike?"** or **"Show me the breakdown by region."**
    
    Make the questions sound like natural curiosity.

    CONSTRAINT:
    - If the answer is not in the context, admit it, but tell them what *is* in the context that might be close.
    GUIDELINES:
    1. **Be Conversational yet Professional:** Write as if you are emailing a client. Don't be robotic.
    2. **Synthesize, Don't Copy:** Do not just paste snippets. Read the context and explain it in your own words.
    3. **Structure Your Answer:**
       - Start with a direct answer (Executive Summary).
       - Use Bullet Points for details.
       - End with a brief conclusion or insight if relevant.
    4. **Citations:** When you use a fact, cite the source number like this [1].
    5. **Honesty:** If the context is missing specific details, politely say so (e.g., "The documents discuss revenue, but do not mention specific Q4 figures.").    
    """
    
    
    user_message = f"""Context Data:
{context_text}

User Query: 
{query}"""

    try:
        # Use Groq's Llama 4 Scout model
        response = get_client().chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Groq API Error: {str(e)}"

def generate_response(query, context):
    """
    Wrapper function for backward compatibility with existing UI code.
    
    Args:
        query: User's question
        context: Retrieved context string
    
    Returns:
        Generated answer
    """
    return generate_answer(query, context)
