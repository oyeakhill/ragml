# Document Q&A Agent with RAG (Retrieval-Augmented Generation)

A production-grade document question-answering system that uses advanced RAG techniques to provide intelligent answers from your PDF documents. This system features enhanced document processing, intelligent retrieval, and answer validation capabilities.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

This Document Q&A Agent is a sophisticated system that allows users to:
- Upload PDF documents
- Ask questions about the content
- Get intelligent, context-aware answers using OpenAI's GPT models
- Manage document lifecycle with ACID-like guarantees

The system uses ChromaDB for vector storage, OpenAI embeddings for semantic search, and includes advanced features like query classification, multi-stage retrieval, and answer validation.

## Features

### Core Features
- **PDF Document Processing**: Advanced PDF parsing with layout preservation
- **Table Extraction**: Automatic detection and extraction of tables from PDFs
- **Intelligent Chunking**: Structure-aware text splitting that preserves document context
- **Semantic Search**: Vector-based similarity search using OpenAI embeddings
- **RAG-powered Q&A**: Context-aware question answering using GPT models

### Enhanced Features
- **Query Classification**: Automatically classifies queries (factual, comparative, analytical, etc.)
- **Multi-stage Retrieval**: Intelligent retrieval strategies based on query type
- **Answer Validation**: Validates answer completeness and confidence
- **Document Management**: ACID-like document lifecycle management with versioning
- **Comparison Handling**: Specialized handling for comparison queries
- **OCR Support**: Optional OCR for image-based PDFs (requires Tesseract)

### UI Features
- **Streamlit Web Interface**: User-friendly web application
- **Real-time Streaming**: Stream answers as they're generated
- **Document Management UI**: Upload, view, and delete documents
- **System Statistics**: Monitor system health and performance

## Architecture

The system follows clean architecture principles with clear separation of concerns:

```
┌─────────────────────┐
│   Streamlit UI      │
│  (enhanced_app.py)  │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Enhanced QA Engine │
│ (Query Processing)  │
└──────────┬──────────┘
           │
     ┌─────┴─────┬─────────┬──────────┐
     │           │         │          │
┌────▼────┐ ┌───▼───┐ ┌──▼───┐ ┌────▼────┐
│Document │ │Vector │ │Query │ │Answer   │
│Manager  │ │Store  │ │Class │ │Validator│
└─────────┘ └───────┘ └──────┘ └─────────┘
```

### Key Components

1. **Document Loader** (`src/enhanced_document_loader.py`)
   - PDF processing with PyMuPDF
   - Table detection and extraction
   - Structure-aware chunking
   - OCR support for scanned PDFs

2. **Document Manager** (`src/document_manager.py`)
   - Repository pattern for document metadata
   - Transactional document operations
   - Content-based deduplication
   - Atomic updates and rollback support

3. **Vector Store** (`src/vector_store.py`)
   - ChromaDB integration
   - OpenAI embeddings
   - Semantic similarity search
   - Batch processing support

4. **Enhanced QA Engine** (`src/enhanced_qa_engine.py`)
   - Query classification
   - Multi-stage retrieval
   - Context assembly
   - Answer generation and validation

## Requirements

### System Requirements
- Python 3.8 or higher
- 4GB+ RAM recommended
- 1GB+ free disk space for vector store

### Python Dependencies
```
openai==1.35.3          # OpenAI API client
streamlit==1.35.0       # Web UI framework
pymupdf==1.24.5         # PDF processing
chromadb==0.5.0         # Vector database
python-dotenv==1.0.1    # Environment management
tiktoken==0.7.0         # Token counting
protobuf==3.20.3        # ChromaDB compatibility
numpy==1.26.4           # Numerical operations
pydantic==2.7.4         # Data validation
tenacity==8.4.1         # Retry logic
```

### Optional Dependencies
```
# For OCR support (optional)
pytesseract             # OCR engine wrapper
pillow                  # Image processing
pdf2image               # PDF to image conversion

# For enhanced features (optional)
langchain==0.2.5        # LLM framework
langchain-openai==0.1.9 # OpenAI integration
langchain-community==0.2.5 # Community tools
```

### External Requirements
- **OpenAI API Key**: Required for embeddings and LLM
- **Tesseract OCR** (optional): For OCR functionality
  - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
  - Linux: `sudo apt-get install tesseract-ocr`
  - macOS: `brew install tesseract`

## Installation

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd ragml
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables
Create a `.env` file in the project root:
```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional - defaults shown
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.7
MAX_TOKENS=2000
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
CHROMA_PERSIST_DIR=./chroma_db
```

### Step 5: Run Setup Script
```bash
python setup.py
```

This will:
- Create necessary directories
- Validate your environment
- Install any missing dependencies

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | - | Yes |
| `EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-3-small` | No |
| `LLM_MODEL` | OpenAI chat model | `gpt-3.5-turbo` | No |
| `LLM_TEMPERATURE` | Response creativity (0-2) | `0.7` | No |
| `MAX_TOKENS` | Max response length | `2000` | No |
| `CHUNK_SIZE` | Document chunk size | `1000` | No |
| `CHUNK_OVERLAP` | Chunk overlap size | `200` | No |
| `TOP_K_RESULTS` | Number of search results | `5` | No |
| `CHROMA_PERSIST_DIR` | Vector store directory | `./chroma_db` | No |

### Configuration Tips
- **For better accuracy**: Use `gpt-4` as `LLM_MODEL`
- **For faster responses**: Reduce `MAX_TOKENS` and `TOP_K_RESULTS`
- **For large documents**: Increase `CHUNK_SIZE` to 1500-2000
- **For technical documents**: Decrease `LLM_TEMPERATURE` to 0.3-0.5

## Usage

### Starting the Application

```bash
# Using Streamlit directly
streamlit run enhanced_app.py

# Or using the wrapper script
python app.py
```

The application will open in your default browser at `http://localhost:8501`

### Basic Workflow

1. **Upload Documents**
   - Click "Upload PDF Document" in the sidebar
   - Select one or more PDF files
   - Wait for processing confirmation

2. **Ask Questions**
   - Type your question in the chat input
   - Enable/disable RAG retrieval with the checkbox
   - Enable answer validation for quality checks

3. **View Results**
   - See the answer with source citations
   - Expand sections for query analysis and validation
   - Check document sources for transparency

4. **Manage Documents**
   - View indexed documents in the sidebar
   - Delete documents with the trash icon
   - Monitor system statistics

### Example Questions

**Factual Questions:**
- "What is the main topic of this document?"
- "What are the key specifications of the GLS model?"

**Comparative Questions:**
- "Compare GLS and GLX models"
- "What's the difference between version 1 and version 2?"

**Analytical Questions:**
- "What are the implications of these findings?"
- "How does this relate to industry standards?"

**Synthesis Questions:**
- "Summarize the key points across all documents"
- "What patterns emerge from the data?"

## Project Structure

```
ragml/
├── .env                    # Environment variables (create this)
├── .gitignore             # Git ignore rules
├── requirements.txt       # Python dependencies
├── setup.py              # Setup script
├── app.py               # Application entry point
├── enhanced_app.py      # Streamlit UI
├── improved_rag.py      # Advanced RAG implementation
│
├── src/                 # Source code
│   ├── __init__.py
│   ├── document_loader.py      # Document loading (legacy)
│   ├── enhanced_document_loader.py  # Enhanced loader
│   ├── document_manager.py     # Document lifecycle
│   ├── qa_engine.py           # QA engine (legacy)
│   ├── enhanced_qa_engine.py  # Enhanced QA engine
│   ├── vector_store.py        # Vector database
│   │
│   └── utils/           # Utilities
│       ├── __init__.py
│       ├── config.py    # Configuration management
│       ├── logger.py    # Logging setup
│       └── text_splitter.py  # Text chunking
│
├── chroma_db/          # Vector store data (auto-created)
├── data/               # Data directory (auto-created)
├── logs/               # Log files (auto-created)
└── uploads/            # Uploaded files (auto-created)
```

## API Documentation

### Core Classes

#### EnhancedQAEngine
Main engine for question answering.

```python
from src.enhanced_qa_engine import EnhancedQAEngine

# Initialize
engine = EnhancedQAEngine()

# Load document
result = engine.load_document("path/to/document.pdf")

# Ask question
answer = engine.answer_question(
    question="What is the main topic?",
    use_rag=True,
    stream=False,
    validate=True
)
```

#### DocumentManager
Manages document lifecycle with ACID guarantees.

```python
from src.document_manager import DocumentManager

# Add document
result = manager.add_document(filename, chunks)

# Delete document
result = manager.delete_document(filename)

# List documents
documents = manager.list_documents()
```

#### VectorStore
Handles embeddings and similarity search.

```python
from src.vector_store import VectorStore

# Initialize
store = VectorStore()

# Add documents
store.add_documents(chunks)

# Search
results = store.search("query", k=5)
```

## Advanced Features

### Query Classification
The system automatically classifies queries into types:
- **Factual**: Direct information retrieval
- **Comparative**: Comparing entities or concepts
- **Analytical**: Analysis and interpretation
- **Synthesis**: Combining information
- **Temporal**: Time-related queries
- **Causal**: Cause-and-effect relationships

### Multi-Stage Retrieval
Based on query type, the system uses different retrieval strategies:
1. Direct semantic search
2. Entity-based retrieval
3. Context expansion for multi-hop queries
4. Re-ranking based on relevance

### Answer Validation
The system validates answers for:
- Completeness
- Confidence level
- Missing information
- Contradictions
- Improvement suggestions

### Document Deduplication
- Content-based hashing prevents duplicate uploads
- Automatic version management
- Atomic updates ensure consistency

## Troubleshooting

### Common Issues

**1. OpenAI API Key Error**
```
Error: OPENAI_API_KEY environment variable is required
```
**Solution**: Create `.env` file with your API key

**2. ChromaDB Initialization Error**
```
Error: Failed to initialize ChromaDB
```
**Solution**: Delete `chroma_db` folder and restart

**3. PDF Processing Error**
```
Error: Failed to process PDF
```
**Solution**: Ensure PDF is not corrupted or password-protected

**4. Memory Error with Large Documents**
```
Error: Out of memory
```
**Solution**: Increase chunk size or process documents in smaller batches

**5. OCR Not Working**
```
Warning: Tesseract OCR not available
```
**Solution**: Install Tesseract and add to PATH

### Performance Optimization

1. **For Large Documents**:
   - Increase `CHUNK_SIZE` to 1500-2000
   - Reduce `CHUNK_OVERLAP` to 100-150
   - Process documents in batches

2. **For Many Documents**:
   - Use document filtering in searches
   - Regularly clean up unused documents
   - Monitor vector store size

3. **For Better Accuracy**:
   - Use GPT-4 model
   - Increase `TOP_K_RESULTS`
   - Enable answer validation

### Debugging

Enable debug logging:
```python
# In src/utils/logger.py
logger = setup_logger("ragml", logging.DEBUG, "debug.log")
```

Check logs in `logs/ragml.log` for detailed information.

## Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests (if available)
5. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and small

### Adding Features

1. **New Document Types**: Extend `DocumentProcessor` in `enhanced_document_loader.py`
2. **New Query Types**: Add to `QueryType` enum in `enhanced_qa_engine.py`
3. **New Retrieval Strategies**: Extend `IntelligentRetriever`
4. **New Validation Rules**: Extend `AnswerValidator`

## License

This project is provided as-is for educational and commercial use. Please ensure you comply with OpenAI's usage policies and any other third-party licenses.

## Acknowledgments

- OpenAI for GPT models and embeddings
- ChromaDB for vector storage
- Streamlit for the web framework
- PyMuPDF for PDF processing
