# 📄 TALKER PDF

A powerful Retrieval-Augmented Generation (RAG) application that allows users to chat with their PDF documents using advanced AI models. Built with Streamlit, ChromaDB, and Groq's Llama models.

## 🌟 Features

- **PDF Document Ingestion**: Upload and process multiple PDF files simultaneously
- **Hybrid Search**: Combines vector search (semantic) and BM25 (keyword-based) for optimal retrieval
- **Advanced Re-ranking**: Uses FlashRank for ultra-fast and accurate result re-ranking
- **Conversational AI**: Powered by Groq's Llama models (Llama 3.1 8B for query contextualization, Llama 4 Scout for answer generation)
- **Context-Aware Chat**: Maintains conversation history and rewrites queries for better understanding
- **Confidence Scoring**: Filters low-confidence results to prevent hallucinations
- **Suggested Questions**: Automatically generates follow-up questions to guide deeper exploration
- **Duplicate Prevention**: Automatically skips already ingested documents
- **Debug Mode**: View retrieved chunks and confidence scores for transparency

## 🏗️ Architecture

```
TALKERPDF/
├── app/
│   └── ui.py                 # Streamlit web interface
├── config/
│   └── settings.py           # Configuration and environment settings
├── core/
│   ├── generation.py         # Answer generation using Groq API
│   ├── ingestion.py          # PDF processing and ChromaDB storage
│   └── retrieval.py          # Hybrid search and re-ranking logic
├── data/
│   ├── chroma_db/            # ChromaDB persistent storage
│   └── docs/                 # Uploaded PDF files
├── requirements.txt          # Python dependencies
└── .gitignore               # Git ignore rules
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Groq API Key (get it from [Groq Console](https://console.groq.com))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/shubham-kr2002/TALKERPDF.git
   cd TALKERPDF
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Run the application**
   ```bash
   streamlit run app/ui.py
   ```

6. **Access the app**
   Open your browser and navigate to `http://localhost:8501`

## 📚 How It Works

### 1. Document Ingestion Pipeline

```python
PDF Upload → Text Extraction → Sliding Window Chunking → Embedding Generation → ChromaDB Storage
```

- **Text Extraction**: Uses PyMuPDF to extract text from PDFs
- **Chunking**: Implements sliding window with 500-character chunks and 100-character overlap
- **Embeddings**: Generates vector embeddings using `all-mpnet-base-v2` model
- **Batch Processing**: Optimized batch encoding for 3-5x faster ingestion

### 2. Retrieval Pipeline

```python
User Query → Query Contextualization → Hybrid Search → RRF Fusion → Re-ranking → Top Results
```

#### Hybrid Search Components:
- **Vector Search**: Semantic similarity using sentence embeddings
- **BM25 Search**: Keyword-based search for exact term matching
- **Reciprocal Rank Fusion (RRF)**: Intelligently merges results from both search methods
- **FlashRank Re-ranking**: Ultra-fast cross-encoder re-ranking (10x faster than traditional methods)

### 3. Answer Generation Pipeline

```python
Query + Context → LLM (Llama 4 Scout) → Structured Answer + Suggested Questions
```

- **Confidence Filtering**: Rejects answers when top chunk confidence < 25%
- **Hallucination Prevention**: Only answers based on retrieved context
- **Citation Support**: References source documents in answers
- **Follow-up Generation**: Suggests relevant next questions

## 🛠️ Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **UI Framework** | Streamlit | Interactive web interface |
| **Vector Database** | ChromaDB | Document storage and vector search |
| **Embeddings** | Sentence-Transformers | Text-to-vector conversion |
| **LLM Provider** | Groq API | Fast inference with Llama models |
| **Re-ranker** | FlashRank | Lightweight and fast result re-ranking |
| **Keyword Search** | BM25Okapi | Traditional keyword-based search |
| **PDF Processing** | PyMuPDF (fitz) | Text extraction from PDFs |

## 📊 Performance Optimizations

1. **Model Caching**: Embedding model and ranker are cached to prevent reloading
2. **Batch Encoding**: Process 32-64 chunks simultaneously during ingestion
3. **BM25 Indexing**: Pre-built keyword index for instant searches
4. **Limited Re-ranking**: Only re-rank top 10 candidates (constant latency)
5. **FlashRank**: 10x faster than traditional cross-encoders

## 🎯 Usage Guide

### Uploading Documents

1. Click the sidebar to expand configuration
2. Enter your Groq API key
3. Upload one or more PDF files
4. Click "Ingest PDFs" to process documents

### Chatting with Documents

1. Type your question in the chat input
2. The system will:
   - Contextualize your query based on chat history
   - Search for relevant chunks using hybrid search
   - Re-rank results for accuracy
   - Generate a comprehensive answer with citations
   - Suggest follow-up questions

### Debug Mode

- Expand the "Debug Retrieval" section under any answer
- View the top 3 retrieved chunks with confidence scores
- See source documents and chunk IDs

## 🔧 Configuration

Edit `config/settings.py` to customize:

```python
# Chunk size and overlap
chunk_size = 500
overlap = 100

# Retrieval parameters
initial_k = 15  # Number of candidates to retrieve
top_k = 5       # Number of final results

# Confidence threshold
confidence_threshold = 0.25
```

## 📝 API Models Used

- **Query Contextualization**: `meta-llama/llama-3.1-8b-instruct:free`
- **Answer Generation**: `meta-llama/llama-4-scout-17b-16e-instruct`
- **Embeddings**: `all-mpnet-base-v2` (384 dimensions)
- **Re-ranking**: `ms-marco-MiniLM-L-12-v2`

## 🔒 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Groq API key for LLM access | Yes |
| `OPENROUTER_API_KEY` | Alternative API key (legacy) | No |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [ChromaDB](https://www.trychroma.com/) for the vector database
- [Groq](https://groq.com/) for blazing-fast LLM inference
- [Sentence-Transformers](https://www.sbert.net/) for embedding models
- [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) for efficient re-ranking

## 📞 Contact

**Shubham Kumar**
- GitHub: [@shubham-kr2002](https://github.com/shubham-kr2002)

## 🐛 Known Issues & Future Improvements

- [ ] Add support for multi-language documents
- [ ] Implement streaming responses for better UX
- [ ] Add support for other document formats (DOCX, TXT, etc.)
- [ ] Implement user authentication and document management
- [ ] Add export conversation history feature
- [ ] Implement semantic caching for repeated queries

---

Made with ❤️ using Streamlit
