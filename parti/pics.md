---
1.
## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT                               │
│              (Chat UI / Query UI / API)                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API SERVER (FastAPI)                         │
│              /api/chat  |  /api/query  |  /api/upload           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              ENHANCED MULTI-AGENT SYSTEM                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              TaskDetector                                │   │
│  │  (Pattern Matching + LLM-based Detection)                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Task Router (Handler Selection)                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   QA Handler │  │ Quiz Handler │  │ Study Guide  │
│              │  │              │  │   Handler    │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │      RAG System (EnhancedRAG)      │
        │  ┌──────────────────────────────┐  │
        │  │  MilvusRetrievalPipeline     │  │
        │  │  (Vector Search + Rerank)    │  │
        │  └──────────────────────────────┘  │
        │  ┌──────────────────────────────┐  │
        │  │  KnowledgeTools              │  │
        │  │  (Hybrid/Multi-Query)        │  │
        │  └──────────────────────────────┘  │
        └────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │         LLM (Gemini)               │
        │    (Answer Generation)             │
        └────────────────────────────────────┘
```




2.
### Quiz Generation Pipeline

```
User Query: "Generate quiz on marine life with 5 questions at medium difficulty"
               │
               ▼
┌─────────────────────────────────────┐
│  _handle_quiz()                     │
│  - Extract: topic, num_questions,   │
│    difficulty                       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  PromptLibrary.get_prompt()         │
│  - Get "difficulty_aware" variant   │
│  - Format with parameters           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  LLM (Gemini)                       │
│  - Generate quiz questions          │
│  - Format: JSON with questions,     │
│    options, answers, explanations   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  ConversationManager                │
│  - Cache quiz for follow-up         │
└──────────────┬──────────────────────┘
               │
               ▼
         Quiz Response
```

3.

## Embedding Pipeline

### Document Indexing (Offline)

```
        Document Upload
                │
                ▼
┌─────────────────────────────────────┐
│  Text Extraction                    │
│  - PDF: PyPDF2/pdfplumber           │
│  - DOCX: python-docx                │
│  - TXT: Direct read                 │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Text Cleaning                      │
│  - Remove extra whitespace          │
│  - Fix encoding issues              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Chunking                           │
│  - Strategy: medium (200-500 tokens)│
│  - Method: content_aware            │
│  - Preserve context                 │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Embedding Generation               │
│  ┌──────────────────────────────┐   │
│  │  Baseline Model              │   │
│  │  (all-mpnet-base-v2)         │   │
│  │  → 768-dim vector            │   │
│  └──────────────────────────────┘   │
│  ┌──────────────────────────────┐   │
│  │  Scientific Model            │   │
│  │  (allenai/scibert_scivocab)  │   │
│  │  → 768-dim vector            │   │
│  └──────────────────────────────┘   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Milvus Indexing                    │
│  - environment_chunks_baseline      │
│  - environment_chunks_scientific    │
│  - Parent-child structure           │
└──────────────┬──────────────────────┘
               │
               ▼
         Indexed & Ready
```

## RAG Pipeline
4.

### Complete RAG Flow

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  EnhancedRAG.answer_question()      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  MilvusRetrievalPipeline            │
│  hierarchical_retrieve()            │
│  ┌──────────────────────────────┐   │
│  │  1. Encode Query (SciBERT)   │   │
│  │  2. Vector Search            │   │
│  │     - Cosine similarity      │   │
│  │     - Top-K chunks           │   │
│  │  3. Rerank (BGE Reranker)    │   │
│  │  4. Context Expansion        │   │
│  │     - Get parent documents   │   │
│  │     - Get sibling chunks     │   │
│  └──────────────────────────────┘   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Evaluate Retrieval Quality         │
│  - Check confidence threshold       │
│  - Check if Milvus available        │
└──────────────┬──────────────────────┘
               │
        ┌──────┴──────┐
        │             │
   Above Threshold  Below Threshold
        │             │
        ▼             ▼
┌──────────────┐  ┌──────────────────┐
│ Use Library  │  │ Web Search        
│ Results      │  │ Fallback          
└──────┬───────┘  └────────┬─────────┘
       │                    │
       └──────────┬─────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Build Context                      │
│  - Library sources (if available)   │
│  - Web sources (if needed)          │
│  - Format: "=== LIBRARY SOURCES === │
│           === WEB SOURCES ==="      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  LLM Prompt Structure               │
│  ┌──────────────────────────────┐   │
│  │ Retrieved Information        │   │
│  │ (Library + Web sources)      │   │
│  ├──────────────────────────────┤   │
│  │ Question                     │   │
│  ├──────────────────────────────┤   │
│  │ Instructions                 │   │
│  │ - Use retrieved info         │   │
│  │ - Reference sources correctly│   │
│  └──────────────────────────────┘   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  LLM (Gemini)                       │
│  - Generate answer                  │
│  - Reference sources                │
└──────────────┬──────────────────────┘
               │
               ▼
         Answer + Sources
```

## Email & Document Generation Pipeline

5.
### Email Sending Flow

```
Content (Quiz/Study Guide/Awareness)
    │
    ▼
┌─────────────────────────────────────┐
│  EmailHandler                       │
│  - Check if document exists         │
│  - Auto-export if needed            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  DocumentTools.generate_document()  │
│  - Format content                   │
│  - Generate PDF/DOCX                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  EmailTools.send_with_retry()       │
│  ┌──────────────────────────────┐   │
│  │ 1. Prepare email             │   │
│  │    - Subject                 │   │
│  │    - Body                    │   │
│  │    - Attachment              │   │
│  ├──────────────────────────────┤   │
│  │ 2. SMTP Connection           │   │
│  │    - Connect to server       │   │
│  │    - Authenticate            │   │
│  ├──────────────────────────────┤   │
│  │ 3. Send with Retry           │   │
│  │    - Max 3 attempts          │   │
│  │    - Exponential backoff     │   │
│  └──────────────────────────────┘   │
└──────────────┬──────────────────────┘
               │
               ▼
         Email Sent Confirmation
``` 
