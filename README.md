# Financial RAG Application For Data Extraction and QA

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technical Pipeline](#technical-pipeline)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Project Overview

The Financial RAG Application is an advanced document processing system that leverages Retrieval Augmented Generation (RAG) technology to intelligently extract structured data from financial documents. The system combines the power of Large Language Models (LLMs) with vector databases to provide context-aware document processing that improves accuracy over time.

### Purpose

Transform unstructured financial documents (invoices, bank statements, receipts) into standardized, structured data formats suitable for accounting systems, ERP integration, and financial analysis.

### Technology Stack

- **Backend**: Python 3.8+, LangChain, Ollama, ChromaDB
- **Frontend**: Streamlit Web Application
- **LLM**: Llama3 (via Ollama for local processing)
- **Vector Database**: ChromaDB for semantic search
- **Document Processing**: PyPDF for PDFs, Tesseract OCR for images
- **Output Generation**: Pandas, OpenPyXL for Excel file creation

## Key Features

### 🧠 **RAG-Powered Intelligence**

- **Context-Aware Extraction**: Uses historical document patterns to improve current extractions
- **Semantic Understanding**: Vector-based similarity matching for document relationships
- **Continuous Learning**: Each processed document enhances future processing accuracy
- **Local LLM Processing**: Privacy-focused with no external API dependencies

### 📄 **Multi-Format Document Support**

- **PDF Documents**: Native text extraction with OCR fallback
- **Image Files**: JPG, JPEG, PNG, TIFF with advanced OCR processing
- **Batch Processing**: Handle multiple documents simultaneously
- **Validation & Quality Control**: Comprehensive error checking and data validation

### 📊 **Dual Excel Output System**

- **Individual Invoices**: Detailed multi-sheet Excel files per document
- **Master Consolidation**: Single Excel file with all invoices as standardized rows
- **Professional Formatting**: Color-coded sections, auto-sizing, and clear organization
- **Template Compliance**: Maps to comprehensive 60+ field accounting template

### 🔍 **Advanced Query Capabilities**

- **Natural Language Search**: Query document corpus in plain English
- **Semantic Search**: Find documents by meaning, not just keywords
- **Pattern Analysis**: Identify trends and anomalies across document sets
- **Source Attribution**: Track which documents contribute to query results

### 🌐 **User-Friendly Interface**

- **Streamlit Web App**: Intuitive drag-and-drop file upload
- **Real-Time Processing**: Live progress tracking and status updates
- **Bulk Download**: ZIP file generation with all processed outputs
- **Health Monitoring**: System status checks and performance metrics

## System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │    │   RAG Pipeline   │    │  Vector Store   │
│   (Streamlit)   │◄──►│   (LangChain)    │◄──►│   (ChromaDB)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         ▼                        ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Document        │    │      LLM         │    │   Embeddings    │
│ Processing      │    │    (Ollama)      │    │ (HuggingFace)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌──────────────────┐
│ Excel Generation│    │ Template Mapping │
│   (Pandas)      │    │   (Accounting)   │
└─────────────────┘    └──────────────────┘
```

### Core Components

#### **1. Document Processing Layer**

- **Purpose**: Convert various document formats into processable text
- **Components**: PyPDFLoader, Tesseract OCR, text chunking
- **Output**: Structured Document objects with metadata

#### **2. Embedding & Vector Layer**

- **Purpose**: Create semantic representations for similarity search
- **Components**: HuggingFace Transformers, ChromaDB
- **Features**: 384-dimensional embeddings, persistent storage

#### **3. RAG Pipeline**

- **Purpose**: Context-aware data extraction using historical patterns
- **Components**: LangChain orchestration, Ollama LLM integration
- **Process**: Retrieval → Context Assembly → Generation

#### **4. Template Mapping System**

- **Purpose**: Standardize extracted data to accounting templates
- **Features**: LLM-powered mapping with rule-based fallbacks
- **Output**: 60+ standardized accounting fields

#### **5. Excel Generation Engine**

- **Purpose**: Create professional Excel outputs
- **Features**: Multi-sheet individual files, consolidated master file
- **Formatting**: Color coding, professional styling, auto-sizing

## Technical Pipeline

### Phase 1: Document Ingestion

1. **File Upload**: Multi-file selection via Streamlit interface
2. **Format Detection**: Automatic identification of PDF vs image files
3. **Text Extraction**:
   - PDFs: PyPDFLoader with metadata preservation
   - Images: Tesseract OCR with preprocessing
4. **Document Chunking**: RecursiveCharacterTextSplitter for optimal processing

### Phase 2: Vector Processing

1. **Embedding Generation**: Convert text chunks to 384-dim vectors
2. **Vector Storage**: Store in ChromaDB with rich metadata
3. **Indexing**: Organize by document type and source information
4. **Similarity Indexing**: Enable fast semantic search capabilities

### Phase 3: RAG Extraction

1. **Context Retrieval**: Find similar historical documents
2. **Context Assembly**: Build relevant pattern examples
3. **Prompt Engineering**: Create context-aware extraction prompts
4. **LLM Processing**: Llama3 processes with historical context
5. **JSON Parsing**: Extract structured data with validation

### Phase 4: Data Standardization

1. **Template Mapping**: Convert to accounting template format
2. **Field Validation**: Check data types and required fields
3. **Quality Control**: Validate extraction accuracy
4. **Error Handling**: Robust fallback mechanisms

### Phase 5: Output Generation

1. **Individual Excel Creation**:
   - Sheet 1: Invoice Summary
   - Sheet 2: Line Items Details
   - Sheet 3: Raw Data
2. **Master Excel Update**: Append to consolidated file
3. **Professional Formatting**: Apply styles and organization
4. **File Management**: Organize outputs by date and type

## Installation & Setup

### Prerequisites

- **Python**: Version 3.8 or higher
- **Ollama**: For local LLM processing
- **Tesseract OCR**: For image document processing
- **System Requirements**: 8GB RAM minimum, 16GB recommended

### Step 1: Environment Setup

```bash
# Clone the repository
git clone
cd financial_rag_app

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### Step 2: Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install additional packages for updated LangChain
pip install -U langchain-chroma langchain-ollama langchain-community pypdf
```

### Step 3: Ollama Setup

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull Llama3 model (in another terminal)
ollama pull llama3

# Verify installation
ollama list
```

### Step 4: Tesseract OCR Installation

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows: Download from GitHub releases
# https://github.com/UB-Mannheim/tesseract/wiki
```

### Step 5: Launch Application

```bash
# Option 1: Using launcher script
python run_app.py

# Option 2: Direct Streamlit command
streamlit run streamlit_app.py

# Access at: http://localhost:8501
```

## Usage Guide

### Basic Workflow

#### 1. **Health Check**

- Click "Health Check" in the sidebar to verify system status
- Ensure all components (Ollama, Vector DB, FileSystem) are healthy

#### 2. **Document Upload**

- Use the file uploader to select multiple invoice files
- Supported formats: PDF, JPG, JPEG, PNG, TIFF
- Review file list and sizes before processing

#### 3. **Process Documents**

- Click "Process Invoices" to start RAG extraction
- Monitor real-time progress bar and status updates
- Review processing results and validation metrics

#### 4. **Download Results**

- **ZIP Download**: All individual Excel files + master file
- **Master Excel**: Consolidated file with all invoices
- Files include professional formatting and multiple sheets

#### 5. **Query Corpus**

- Use natural language to search processed documents
- Examples: "Find invoices over $10,000", "Show GST details"
- Review results with source attribution

### Advanced Features

#### **Corpus Management**

- **Statistics**: View total documents, chunks, and file types
- **Clear Database**: Reset vector database for fresh start
- **Pattern Analysis**: Discover trends across document types

#### **Quality Control**

- **Validation Results**: Review extraction accuracy warnings
- **Error Reports**: Detailed information on processing failures
- **Success Metrics**: Track processing success rates

#### **Batch Operations**

- **Folder Processing**: Process entire directories of documents
- **Progress Tracking**: Monitor batch processing status
- **Summary Reports**: Comprehensive batch processing statistics

## API Reference

### Core Classes

#### **FinancialRAGApp**

Main application class that orchestrates the entire pipeline.

**Key Methods:**

- `process_document(file_path)`: Process single document
- `process_batch(directory_path)`: Process multiple documents
- `query_documents(query, document_type)`: Semantic search
- `get_corpus_stats()`: Retrieve system statistics
- `health_check()`: System health verification

#### **RAGPipeline**

Handles the core RAG processing workflow.

**Key Methods:**

- `extract_invoice_data_with_rag(file_path)`: Context-aware extraction
- `query_document_corpus(query)`: Query indexed documents
- `find_similar_documents(reference_file)`: Find similar documents
- `analyze_document_patterns()`: Pattern analysis

#### **VectorStoreManager**

Manages ChromaDB operations and semantic search.

**Key Methods:**

- `add_documents(documents)`: Add documents to vector store
- `similarity_search(query)`: Perform semantic search
- `get_relevant_context(query)`: Retrieve contextual information
- `clear_collection()`: Clear all stored documents

### Data Structures

#### **Invoice Data Schema**

```json
{
  "invoice_info": {
    "invoice_number": "string",
    "invoice_date": "YYYY-MM-DD",
    "due_date": "YYYY-MM-DD",
    "po_number": "string",
    "currency": "string"
  },
  "vendor": {
    "name": "string",
    "address": "string",
    "phone": "string",
    "email": "string",
    "gst_number": "string",
    "pan": "string"
  },
  "customer": {
    "name": "string",
    "address": "string",
    "phone": "string",
    "email": "string",
    "gst_number": "string",
    "pan": "string"
  },
  "line_items": [
    {
      "item_name": "string",
      "item_code": "string",
      "hsn_sac_code": "string",
      "quantity": "number",
      "taxable_amount": "number",
      "discount_amount": "number",
      "cgst_rate": "number",
      "cgst_amount": "number",
      "sgst_rate": "number",
      "sgst_amount": "number",
      "igst_rate": "number",
      "igst_amount": "number",
      "cess_rate": "number",
      "cess_amount": "number",
      "total_amount": "number"
    }
  ],
  "totals": {
    "tax_total": "number",
    "grand_total": "number"
  },
  "payment_info": {
    "payment_terms": "string",
    "payment_method": "string",
    "bank_account": "string",
    "ifsc_code": "string"
  }
}
```

## Configuration

### Environment Variables

Configure the application using `.env` file:

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# Vector Database
VECTOR_DB_TYPE=chroma
CHROMA_PERSIST_DIR=output/vector_db/chroma

# Logging
LOG_LEVEL=INFO
LOG_FILE=output/logs/app.log

# Optional: OpenAI API Key (if using OpenAI embeddings)
# OPENAI_API_KEY=your_key_here
```

### Application Settings

Key configuration parameters in `src/config.py`:

- **CHUNK_SIZE**: Text chunk size for processing (default: 1000)
- **CHUNK_OVERLAP**: Overlap between chunks (default: 200)
- **MAX_RELEVANT_CHUNKS**: Maximum context chunks (default: 5)
- **SIMILARITY_THRESHOLD**: Minimum similarity score (default: 0.7)

### Model Configuration

- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **LLM Model**: Llama3 via Ollama
- **Vector Dimensions**: 384
- **Temperature**: 0.1 (for consistent extractions)

## Troubleshooting

### Common Issues

#### **Ollama Connection Errors**

**Symptoms**: "unhealthy" status in health check, LLM processing failures
**Solutions**:

- Verify Ollama is running: `ollama serve`
- Check model availability: `ollama list`
- Restart Ollama service if needed

#### **Vector Database Issues**

**Symptoms**: ChromaDB errors, indexing failures
**Solutions**:

- Clear vector database using sidebar button
- Check disk space in `output/vector_db/` directory
- Restart application to reinitialize database

#### **PDF Processing Failures**

**Symptoms**: "pypdf package not found", empty document extraction
**Solutions**:

- Install/update pypdf: `pip install -U pypdf`
- Verify PDF file integrity
- Check file permissions and accessibility

#### **OCR Processing Issues**

**Symptoms**: Poor text extraction from images, OCR errors
**Solutions**:

- Verify Tesseract installation: `tesseract --version`
- Check image quality and resolution
- Ensure supported image formats (JPG, PNG, TIFF)

#### **Memory Issues**

**Symptoms**: Out of memory errors, slow processing
**Solutions**:

- Process documents in smaller batches
- Increase system RAM or use swap space
- Clear vector database periodically

#### **Excel Generation Errors**

**Symptoms**: File creation failures, formatting issues
**Solutions**:

- Verify output directory permissions
- Check available disk space
- Ensure Excel files aren't open in other applications

### Performance Optimization

#### **Processing Speed**

- Use PDF documents when possible (faster than OCR)
- Process documents in batches of 10-20 for optimal performance
- Clear vector database periodically to maintain speed

#### **Memory Management**

- Monitor system memory usage during batch processing
- Restart application after processing large document sets
- Use smaller chunk sizes for memory-constrained systems

#### **Storage Optimization**

- Regularly backup and archive old vector databases
- Compress output Excel files for storage
- Clean temporary files from processing directories

### Debugging

#### **Enable Debug Logging**

Set `LOG_LEVEL=DEBUG` in configuration to get detailed logs.

#### **Health Check Details**

Use the health check feature to identify specific component issues:

- **Ollama**: Connection and model availability
- **Vector DB**: Database access and statistics
- **FileSystem**: Directory permissions and space

#### **Error Analysis**

Check logs in `output/logs/app.log` for detailed error information and stack traces.

## Contributing

### Development Setup

1. Fork the repository and create a feature branch
2. Follow the installation guide for development environment
3. Install additional development dependencies
4. Run tests to ensure functionality

### Code Standards

- **Python Style**: Follow PEP 8 guidelines
- **Documentation**: Include docstrings for all functions
- **Type Hints**: Use type annotations throughout
- **Error Handling**: Implement comprehensive error handling

### Testing Guidelines

- Unit tests for core components
- Integration tests for complete workflows
- Performance tests for large document batches
- User acceptance tests for UI functionality

### Contribution Workflow

1. **Issue Creation**: Create detailed issue descriptions
2. **Feature Development**: Implement with tests and documentation
3. **Pull Request**: Submit with clear description and test results
4. **Code Review**: Address feedback and suggestions
5. **Merge**: Maintainers will merge approved contributions

### Feature Requests

Priority areas for enhancement:

- **New Document Types**: Bank statements, receipts, purchase orders
- **Additional LLM Support**: Integration with other local LLMs
- **Enhanced Analytics**: Advanced pattern recognition and reporting
- **API Development**: REST API for programmatic access
- **Mobile Support**: Responsive design for mobile devices

## License

[Include appropriate license information]

## Support

For support, questions, or contributions, please [include contact information or repository links].

## Changelog

[Maintain version history and feature updates]

_This documentation is maintained by the Financial RAG Application development team. Last updated: [Date]_
