# Environmental Sustainability SME: Architecture & Workflow Documentation

## Table of Contents
1. [Overview: How This Project is an SME](#overview)
2. [Complete Workflow](#workflow)
3. [Component Architecture](#architecture)
4. [Strategies & Implementation](#strategies)
5. [Tech Stack](#tech-stack)

---

## 1. Overview: How This Project is an SME {#overview}

### What is an SME (Subject Matter Expert)?
An SME is a specialized AI system that has deep knowledge in a specific domain (Environmental Sustainability) and can:
- Answer domain-specific questions accurately
- Generate educational content (quizzes, study guides)
- Provide expert-level insights
- Adapt responses based on retrieved knowledge
- Handle complex multi-step tasks

### How This System Functions as an SME:

**1. Domain-Specific Knowledge Base:**
- **6,865+ indexed chunks** from environmental/sustainability documents
- **40+ processed documents** covering climate change, renewable energy, conservation, etc.
- **Specialized embeddings** using SciBERT for scientific environmental content

**2. Multi-Agent Task Orchestration:**
- **TaskDetector**: Intelligently identifies user intent (QA, quiz generation, email, etc.)
- **EnhancedMultiAgent**: Routes tasks to specialized handlers
- **Context-Aware Processing**: Maintains conversation history for follow-up requests

**3. Expert-Level Capabilities:**
- **RAG (Retrieval-Augmented Generation)**: Answers grounded in retrieved knowledge
- **Hybrid Retrieval**: Combines dense vector search + keyword-based (BM25) search
- **Web Search Fallback**: Fills gaps when local knowledge is insufficient
- **Content Generation**: Creates quizzes, study guides, awareness content

**4. Intelligent Task Handling:**
- **Multi-step reasoning**: "Generate quiz → Export to PDF → Send via email"
- **Context awareness**: "Send it to my email" (knows what "it" refers to)
- **Format adaptation**: PDF, DOCX, PPTX export based on user preference

---

## 2. Complete Workflow {#workflow}

### High-Level Flow Diagram:
```
User Query
    ↓
[Input Guardrails] → Sanitize & Validate
    ↓
[TaskDetector] → Identify Task Type (QA/Quiz/Email/etc.)
    ↓
[EnhancedMultiAgent] → Route to Handler
    ↓
┌─────────────────────────────────────────┐
│  Handler Execution:                      │
│  - QA Handler → RAG System              │
│  - Quiz Handler → Content Generator      │
│  - Email Handler → Document Export + Email│
│  - Combined Handler → Multi-step Tasks   │
└─────────────────────────────────────────┘
    ↓
[Knowledge Retrieval] → Vector DB + Web Search
    ↓
[LLM Generation] → Gemini 2.0 Flash
    ↓
[Output Guardrails] → Content Moderation
    ↓
Response to User
```

### Detailed Workflow Steps:

#### **Step 1: User Input Processing**
1. **Input Guardrails** (`parti/guardrails.py`):
   - Sanitizes user input (removes HTML/scripts)
   - Detects prompt injection attempts
   - Validates input length (max 5000 chars)
   - Logs suspicious patterns

2. **Task Detection** (`parte/enhanced_multi_agent.py`):
   - **Pattern Matching**: Fast rule-based detection
   - **LLM-Based Detection**: For ambiguous queries
   - **Task Types**: QA, Quiz, Study Guide, Email, Document Export, Library Upload, Awareness Email, Combined

#### **Step 2: Task Routing**
- **TaskRouter** (`parte/task_router.py`):
  - Routes to appropriate handler based on task type
  - Extracts parameters (topic, difficulty, email addresses, format)
  - Maintains routing statistics

#### **Step 3: Handler Execution**

**A. QA Handler:**
```
User Question → RAG System → Knowledge Retrieval → LLM Answer Generation
```

**B. Quiz/Study Guide Handler:**
```
User Request → Content Generator → RAG for Context → LLM Generation → Caching
```

**C. Email Handler:**
```
User Request → Check Cached Content → Export Document → Send Email
```

**D. Combined Handler:**
```
Multi-step Request → Sort Subtasks → Execute Sequentially → Aggregate Results
```

#### **Step 4: Knowledge Retrieval (RAG)**

**Retrieval Pipeline:**
```
Query → Query Expansion → Vector Search (Milvus) → Reranking → Context Building
```

1. **Query Processing** (`partg/query_processor.py`):
   - Expands query with synonyms
   - Generates multiple query variants
   - Enhances for better retrieval

2. **Vector Search** (`partc/milvus_retrieval_pipeline.py`):
   - Encodes query using embedding model
   - Searches Milvus vector database
   - Retrieves top-k candidates (default: 10)

3. **Reranking** (`partc/reranker.py`):
   - Uses BGE Reranker (cross-encoder)
   - Re-scores candidates for relevance
   - Falls back to cosine similarity if BGE unavailable

4. **Context Expansion**:
   - Retrieves parent document metadata
   - Adds contextual chunks from same document
   - Expands to 5 final results

5. **Web Search Fallback** (`partd/enhanced_rag.py`):
   - If local results below threshold (0.3)
   - Performs DuckDuckGo web search
   - Cleans and validates web results

#### **Step 5: LLM Answer Generation**

**Prompt Construction:**
```
SYSTEM: You are an Environmental Sustainability expert
CONTEXT: Retrieved chunks from library + web sources
QUESTION: User's question
INSTRUCTIONS: Use retrieved information, cite sources correctly
```

**Generation** (`partd/llm_client.py`):
- Uses Gemini 2.0 Flash model
- Temperature: 0.7 (balanced creativity)
- Max tokens: 2048
- Multi-step chain generation for complex tasks

#### **Step 6: Output Processing**
1. **Output Guardrails**:
   - Detects harmful content
   - Filters profanity (logs warnings)
   - Blocks dangerous instructions
   - Validates output length

2. **Response Formatting**:
   - Structures response with sources
   - Includes metadata (confidence, retrieval method)
   - Formats for UI display

---

## 3. Component Architecture {#architecture}

### Core Components:

#### **1. Enhanced Multi-Agent System** (`parte/enhanced_multi_agent.py`)
- **Purpose**: Central orchestrator for all tasks
- **Components**:
  - `TaskDetector`: Identifies task type
  - `ConversationManager`: Maintains context
  - `ObservationsLogger`: Tracks performance
  - `PromptLibrary`: Manages prompts

#### **2. LLM Client** (`partd/llm_client.py`)
- **Model**: Google Gemini 2.0 Flash
- **Capabilities**:
  - Text generation
  - JSON generation (for structured outputs)
  - Multi-step chain generation
  - Context-aware generation

#### **3. RAG System** (`partd/enhanced_rag.py`)
- **Components**:
  - `EnhancedRAG`: Main RAG orchestrator
  - `MilvusRetrievalPipeline`: Vector search
  - `BGEReranker`: Relevance reranking
  - Web search integration

#### **4. Knowledge Tools** (`parth/knowledge_tools.py`)
- **Methods**:
  - `retrieve_knowledge()`: Hybrid retrieval
  - `_rag_retrieve()`: RAG-based retrieval
  - `_hybrid_retrieve()`: Dense + sparse search
  - `_multi_query_retrieve()`: Multi-query expansion
  - `_apply_similarity_cutoff()`: Filters by score threshold

#### **5. Task Router** (`parte/task_router.py`)
- **Routing Strategies**:
  - Rule-based (pattern matching)
  - LLM-based (intelligent routing)
  - Combined (hybrid approach)

#### **6. Content Generator** (`partd/task_handlers.py`)
- **Capabilities**:
  - Quiz generation
  - Study guide generation
  - Awareness content generation
  - Uses RAG for context

#### **7. Email Tools** (`parth/email_tools.py`, `partd/email_tool.py`)
- **Features**:
  - Single email sending
  - Batch email campaigns
  - Document attachments (PDF, DOCX, PPTX)
  - Retry logic with exponential backoff

---

## 4. Strategies & Implementation {#strategies}

### 4.1 Chunking Strategy

**Multi-Strategy Approach** (`partb/text_chunker.py`):

#### **Three Chunking Methods:**

1. **Fixed-Size Chunking**:
   - Splits text at fixed token boundaries
   - Preserves exact token counts
   - Use case: Consistent model inputs

2. **Content-Aware Chunking**:
   - Respects paragraph/section boundaries
   - Preserves document structure
   - Use case: Context integrity

3. **Recursive Character Splitting**:
   - Splits at sentence boundaries
   - Maintains readability
   - Use case: Natural language processing

#### **Three Granularities:**

- **Large (2048 tokens)**: Full context for complex reasoning
- **Medium (512 tokens)**: Balanced for general tasks (DEFAULT)
- **Small (128 tokens)**: Focused for specific facts

#### **Overlap Strategy:**
- **Large chunks**: 10% overlap
- **Medium chunks**: 15% overlap
- **Small chunks**: 20% overlap

**Rationale**: Overlap ensures context continuity across chunk boundaries, preventing information loss at split points.

**Tech Stack**: Python `text_chunker.py`, regex for boundary detection, token estimation (~4 chars/token)

---

### 4.2 Embedding Strategy

**Multi-Model Approach** (`partc/embedding_generator.py`):

#### **Embedding Models:**

1. **Baseline Model**: `all-mpnet-base-v2`
   - **Dimensions**: 768
   - **Use Case**: General-purpose text understanding
   - **Performance**: SOTA on semantic similarity benchmarks

2. **Scientific Model**: `allenai/scibert_scivocab_uncased`
   - **Dimensions**: 768
   - **Use Case**: Academic environmental literature, research papers
   - **Training**: 1.14M scientific papers including environmental science
   - **Priority**: Preferred for environment domain

3. **Climate Model** (Optional): `climatebert/distilroberta-base-climate-f`
   - **Dimensions**: 768
   - **Use Case**: Climate change specific topics

4. **Biomedical Model** (Optional): `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
   - **Dimensions**: 768
   - **Use Case**: Environmental health topics

#### **Embedding Generation Process:**
1. Load chunked documents
2. Generate embeddings for each chunk using selected models
3. Normalize embeddings (L2 normalization for cosine similarity)
4. Store with metadata (chunk ID, source file, model used)

**Tech Stack**: 
- `sentence-transformers` library
- PyTorch (CUDA if available)
- Batch processing (32 chunks/batch)

---

### 4.3 Indexing Strategy

**Milvus Vector Database** (`partc/vector_indexer_milvus.py`):

#### **Collection Structure:**

1. **Chunk Collections**:
   - `environment_chunks_baseline`: Baseline embeddings
   - `environment_chunks_scientific`: SciBERT embeddings
   - **Schema**: 
     - `chunk_id` (VARCHAR, primary key)
     - `text` (VARCHAR, chunk content)
     - `embedding` (FLOAT_VECTOR, 768 dimensions)
     - `source_file` (VARCHAR)
     - `chunk_index` (INT64)
     - `metadata` (JSON)

2. **Parent Document Collection**:
   - `environment_documents`: Document-level metadata
   - **Schema**:
     - `document_id` (VARCHAR, primary key)
     - `filename` (VARCHAR)
     - `title` (VARCHAR)
     - `metadata` (JSON)
     - `dummy_vector` (FLOAT_VECTOR, 1 dim) - Required by Milvus 2.6+

#### **Indexing Process:**
1. Load embedding files (JSON format)
2. Create collections if not exist
3. Insert chunks with embeddings
4. Create vector index (IVF_FLAT or HNSW)
5. Link chunks to parent documents

**Tech Stack**:
- Milvus Lite (embedded database)
- pymilvus 2.6+ (Python client)
- JSON for metadata storage

---

### 4.4 Retrieval Strategy

**Hierarchical Retrieval** (`partc/milvus_retrieval_pipeline.py`):

#### **Retrieval Pipeline:**

1. **Query Encoding**:
   - Encode query using same embedding model as chunks
   - Use scientific model (SciBERT) if available

2. **Vector Search**:
   - Search Milvus collection
   - Retrieve top-k candidates (default: 10)
   - Use cosine similarity (normalized embeddings)

3. **Reranking**:
   - Apply BGE Reranker to top candidates
   - Re-score based on query-document relevance
   - Select top 5 after reranking

4. **Context Expansion**:
   - Retrieve parent document metadata
   - Add contextual chunks from same document
   - Expand to final result set

#### **Retrieval Methods** (`parth/knowledge_tools.py`):

1. **RAG Retrieve**:
   - Direct vector search
   - Reranking
   - LLM answer generation

2. **Hybrid Retrieve**:
   - Combines dense (vector) + sparse (BM25) search
   - Weighted combination (alpha=0.6 for dense)
   - Reranking on combined results

3. **Multi-Query Retrieve**:
   - Expands query to multiple variants
   - Retrieves for each variant
   - Aggregates and deduplicates results

**Tech Stack**:
- Milvus for vector search
- BGE Reranker (cross-encoder)
- NumPy for similarity calculations

---

### 4.5 Reranking Strategy

**BGE Reranker with Fallback** (`partc/reranker.py`):

#### **Primary Method: BGE Reranker**
- **Model**: `BAAI/bge-reranker-base`
- **Type**: Cross-encoder (query-document pairs)
- **Process**:
  1. Create query-document pairs
  2. Score each pair using cross-encoder
  3. Sort by rerank score
  4. Return top-k results

#### **Fallback: Cosine Similarity**
- If BGE unavailable, use cosine similarity
- Requires query embedding
- Less accurate but faster

#### **Metrics Calculation**:
- Average rerank score
- Average original score
- Score improvement percentage

**Tech Stack**:
- `sentence-transformers` CrossEncoder
- PyTorch
- NumPy for cosine similarity

---

### 4.6 RAG Strategy

**Enhanced RAG with Web Fallback** (`partd/enhanced_rag.py`):

#### **RAG Pipeline:**

1. **Local Retrieval**:
   - Query Milvus vector database
   - Retrieve top-k chunks
   - Evaluate quality (score threshold: 0.3)

2. **Quality Evaluation**:
   - Check if top result score >= threshold
   - Verify Milvus availability
   - Assess result relevance

3. **Web Search Fallback**:
   - Triggered if local results below threshold
   - Uses DuckDuckGo search API
   - Cleans and validates results
   - Filters irrelevant results

4. **Context Building**:
   - Separates library sources from web sources
   - Numbers sources distinctly
   - Creates source summary
   - Provides clear instructions to LLM

5. **Answer Generation**:
   - Constructs prompt with retrieved context
   - Includes source referencing instructions
   - Generates answer using Gemini
   - Cites sources correctly

#### **Self-Learning (Bonus)**:
- Stores feedback on answer quality
- Uses feedback to improve future answers
- Tracks successful retrieval patterns

**Tech Stack**:
- Milvus for local retrieval
- DuckDuckGo Search API
- Gemini 2.0 Flash for generation

---

### 4.7 Search Strategy

**Hybrid Search** (`parth/knowledge_tools.py`):

#### **Dense Search (Vector)**:
- Semantic similarity using embeddings
- Captures meaning and context
- Alpha weight: 0.6

#### **Sparse Search (BM25)**:
- Keyword-based matching
- Captures exact term matches
- Alpha weight: 0.4 (implicit)

#### **Combination**:
- Weighted sum of scores
- Normalizes both score types
- Reranks combined results

**Tech Stack**:
- Milvus for dense search
- BM25 implementation (if available)
- NumPy for score combination

---

### 4.8 Email Handling Strategy

**Multi-Format Email Support** (`parth/email_tools.py`, `partd/email_tool.py`):

#### **Email Types:**

1. **Single Email**:
   - Send document to one recipient
   - Supports PDF, DOCX, PPTX attachments
   - Custom subject and body

2. **Batch Email**:
   - Send to multiple recipients
   - Awareness campaign support
   - Retry logic for failed sends

#### **Email Process:**

1. **Content Generation** (if needed):
   - Generate awareness content
   - Create quiz/study guide
   - Format for email

2. **Document Export**:
   - Export to requested format (PDF/DOCX/PPTX)
   - Verify file creation
   - Check file size

3. **Email Composition**:
   - Create MIME multipart message
   - Set proper Content-Type headers
   - Attach documents

4. **Sending**:
   - SMTP connection (Gmail: smtp.gmail.com:587)
   - TLS encryption
   - Retry with exponential backoff (3 attempts)

#### **Error Handling**:
- File existence verification
- File size validation
- Attachment error logging
- Graceful failure handling

**Tech Stack**:
- Python `smtplib`
- `email.mime` for message construction
- Gmail SMTP (configurable)

---

### 4.9 Model Evaluation Strategy

**Evaluation Approaches:**

#### **1. Retrieval Quality Metrics:**
- **Top Result Score**: Confidence in best match
- **Average Score**: Overall retrieval quality
- **Rerank Improvement**: Score change after reranking
- **Threshold-Based**: Results above/below confidence threshold

#### **2. Answer Quality Assessment:**
- **Source Citation**: Correct source referencing
- **Relevance**: Answer matches question
- **Completeness**: Covers all aspects
- **Accuracy**: Factual correctness (manual review)

#### **3. System Performance:**
- **Retrieval Speed**: Time to retrieve chunks
- **Generation Speed**: LLM response time
- **Success Rate**: Percentage of successful queries
- **Fallback Rate**: Frequency of web search usage

#### **4. User Feedback (Bonus)**:
- Stores feedback on answer quality
- Tracks successful patterns
- Improves future responses

**Tech Stack**:
- Logging for metrics
- JSON storage for feedback
- Manual evaluation for accuracy

---

## 5. Tech Stack {#tech-stack}

### **Core Technologies:**

#### **LLM & AI:**
- **Google Gemini 2.0 Flash**: Primary LLM for generation
- **Sentence Transformers**: Embedding models
- **BGE Reranker**: Cross-encoder for reranking
- **PyTorch**: Deep learning framework

#### **Vector Database:**
- **Milvus Lite**: Embedded vector database
- **pymilvus 2.6+**: Python client
- **Collections**: Chunk and document storage

#### **Embedding Models:**
- **all-mpnet-base-v2**: General-purpose baseline
- **allenai/scibert_scivocab_uncased**: Scientific text (preferred)
- **climatebert/distilroberta-base-climate-f**: Climate-specific (optional)

#### **Web Framework:**
- **FastAPI**: REST API server
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation

#### **Document Processing:**
- **PyPDF2/pdfplumber**: PDF extraction
- **python-docx**: DOCX processing
- **ReportLab**: PDF generation
- **python-pptx**: PPTX generation

#### **Email:**
- **smtplib**: SMTP email sending
- **email.mime**: Email message construction

#### **Search:**
- **DuckDuckGo Search API**: Web search fallback
- **BM25**: Keyword-based search (if available)

#### **Data Processing:**
- **NumPy**: Numerical operations
- **JSON**: Data serialization
- **Pathlib**: File system operations

#### **Logging & Monitoring:**
- **Python logging**: Application logging
- **Observations Logger**: Performance tracking

### **Architecture Patterns:**

1. **Multi-Agent System**: Task-based routing and execution
2. **RAG (Retrieval-Augmented Generation)**: Knowledge-grounded responses
3. **Hybrid Retrieval**: Dense + sparse search combination
4. **Hierarchical Retrieval**: Multi-granularity chunk access
5. **Chain-of-Thought**: Multi-step reasoning for complex tasks
6. **Context-Aware Processing**: Conversation history management

### **Data Flow:**

```
Documents → Text Extraction → Cleaning → Chunking → Embedding → Indexing → Retrieval → Reranking → LLM Generation → Response
```

### **Key Design Decisions:**

1. **SciBERT Priority**: Prefer scientific embeddings for environment domain
2. **Web Search Fallback**: Ensures answers even when local knowledge insufficient
3. **Multi-Model Embeddings**: Baseline + scientific for comprehensive coverage
4. **Strict Cutoff Enforcement**: Never returns library results below similarity threshold
5. **Non-Blocking Guardrails**: Security measures don't break functionality
6. **Context Caching**: Enables follow-up requests ("send it to my email")

---

## Summary

This Environmental Sustainability SME combines:
- **Domain Expertise**: 6,865+ indexed chunks from environmental documents
- **Intelligent Routing**: Multi-agent system for task orchestration
- **Advanced Retrieval**: Hybrid search with reranking
- **Robust Generation**: Gemini-powered content creation
- **Security**: Input/output guardrails
- **Flexibility**: Multiple export formats, email support, batch operations

The system is production-ready with error handling, logging, fallbacks, and security measures while maintaining high-quality, domain-specific responses.


