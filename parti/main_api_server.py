"""
Main API Server - Part I
FastAPI-based REST API for the Environmental SME System
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import uvicorn
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Ensure Part H modules that rely on relative imports can be resolved
parth_dir = project_root / "parth"
if str(parth_dir) not in sys.path:
    sys.path.append(str(parth_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app (SINGLE INSTANCE)
app = FastAPI(
    title="Environmental Sustainability SME API",
    description="Complete RAG-based AI system for Environmental Sustainability Education",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

# Save the web UI HTML if it doesn't exist
index_html = static_dir / "index.html"
if not index_html.exists():
    logger.info("Creating default index.html...")
    # You need to copy your web_ui.html content here or create a simple placeholder
    index_html.write_text("""
    <!DOCTYPE html>
    <html><head><title>Environmental SME</title></head>
    <body><h1>Environmental Sustainability SME System</h1>
    <p>API is running. Visit <a href="/docs">/docs</a> for API documentation.</p>
    </body></html>
    """)

# Mount static files
try:
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"✓ Static files mounted from {static_dir}")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

# Import pipeline components
try:
    from parth.pipeline_integrator import PipelineIntegrator
except ImportError as e:
    logger.error(f"Failed to import PipelineIntegrator: {e}")
    PipelineIntegrator = None

from parti.pipeline_manager import PipelineManager

# Initialize pipeline + pipeline manager
pipeline = None
pipeline_manager = PipelineManager(project_root=project_root)

# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    method: str = Field("hybrid", description="Retrieval method")
    top_k: int = Field(5, description="Number of results")
    similarity_cutoff: Optional[float] = Field(None, description="Similarity score cutoff for library results (0-1). If set, only library results above this score are shown, remaining slots filled with web results.")

class QuizRequest(BaseModel):
    topic: str = Field(..., description="Quiz topic")
    num_questions: int = Field(5, description="Number of questions")
    difficulty: str = Field("medium", description="Difficulty level")

class StudyGuideRequest(BaseModel):
    topic: str = Field(..., description="Study guide topic")

class DocumentExportRequest(BaseModel):
    content: Dict[str, Any] = Field(..., description="Content to export")
    content_type: str = Field(..., description="Content type")
    format: str = Field("pdf", description="Export format")

class EmailRequest(BaseModel):
    recipient: str = Field(..., description="Recipient email")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body")
    attachments: Optional[List[str]] = Field(None, description="Attachment paths")

class SendEmailRequest(BaseModel):
    recipient: str = Field(..., description="Recipient email")
    content: Dict[str, Any] = Field(..., description="Generated content (quiz or study guide)")
    content_type: str = Field(..., description="Content type: quiz or study-guide")
    topic: str = Field(..., description="Topic of the content")
    format: str = Field("pdf", description="Document format: pdf, docx, or ppt")

class AwarenessEmailRequest(BaseModel):
    topic: str = Field(..., description="Topic for awareness email")
    audience: str = Field(..., description="Target audience type")
    recipients: List[str] = Field(..., description="List of recipient email addresses")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID")

class PipelineActionRequest(BaseModel):
    action: str = Field(..., description="Pipeline action identifier (e.g., load_data)")
    payload: Optional[Dict[str, Any]] = Field(None, description="Optional JSON payload")
@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    global pipeline
    logger.info("Starting API server...")
    
    # Check for required environment variables
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("⚠ GEMINI_API_KEY not set in environment")
        logger.warning("Components will be initialized on first request if API key is available")
    else:
        logger.info("✓ GEMINI_API_KEY found in environment")
    
    # Check for email credentials
    sender_email = os.getenv("SENDER_EMAIL", "sethharry123@gmail.com")
    sender_password = os.getenv("SENDER_PASSWORD", "jkex gegg zrhm ryig")
    if sender_email and sender_password:
        os.environ["SENDER_EMAIL"] = sender_email
        os.environ["SENDER_PASSWORD"] = sender_password
        logger.info("✓ Email credentials configured")
    else:
        logger.warning("⚠ Email credentials not set - email functionality may be limited")
    
    if PipelineIntegrator is None:
        logger.error("PipelineIntegrator not available - will use direct initialization")
        # Try direct initialization
        _ensure_pipeline_components()
        return
    
    try:
        # Try milvus_lite.db first (more recent), fallback to milvus_data.db
        milvus_path = project_root / 'partc' / 'milvus_lite.db'
        if not milvus_path.exists():
            milvus_path = project_root / 'partc' / 'milvus_data.db'
        logger.info(f"Using Milvus database: {milvus_path}")
        config = {
            'milvus_db_path': str(milvus_path)
        }
        logger.info("Attempting to initialize PipelineIntegrator...")
        pipeline = PipelineIntegrator(config)
        logger.info("✓ Pipeline initialized successfully")
        
        # Verify critical components
        if hasattr(pipeline, 'llm') and pipeline.llm:
            logger.info("✓ LLM client available")
        else:
            logger.warning("⚠ LLM client not available")
            
        if hasattr(pipeline, 'rag') and pipeline.rag:
            logger.info("✓ RAG system available")
        else:
            logger.warning("⚠ RAG system not available")
            
        if hasattr(pipeline, 'content_generator') and pipeline.content_generator:
            logger.info("✓ Content generator available")
        else:
            logger.warning("⚠ Content generator not available - will initialize on demand")
            
        if hasattr(pipeline, 'agent') and pipeline.agent:
            logger.info("✓ Agent available")
        else:
            logger.warning("⚠ Agent not available, but content generator can be used directly")
        
        # CRITICAL: Initialize enhanced_agent if pipeline was created via PipelineIntegrator
        # PipelineIntegrator doesn't create enhanced_agent, so we need to add it
        if not hasattr(pipeline, 'enhanced_agent') or pipeline.enhanced_agent is None:
            logger.info("PipelineIntegrator doesn't have enhanced_agent - initializing it now...")
            try:
                _ensure_pipeline_components()
                if hasattr(pipeline, 'enhanced_agent') and pipeline.enhanced_agent is not None:
                    logger.info("✓ Enhanced agent initialized and added to pipeline")
                else:
                    logger.warning("⚠ Enhanced agent initialization failed")
            except Exception as e:
                logger.error(f"Failed to initialize enhanced agent in startup: {e}")
                import traceback
                logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.warning("Server will initialize components on first request")
        pipeline = None

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Environmental Sustainability SME API",
        "version": "1.0.0",
        "status": "running",
        "pipeline_status": "initialized" if pipeline else "not initialized",
        "pipeline_actions": [
            "load_data",
            "build_index",
            "core_sme",
            "agent_suite",
            "prompt_lab",
            "hybrid_rag",
            "full_pipeline",
        ],
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "ui": "/static/index.html",
            "chat": "/api/chat",
            "query": "/api/query",
            "quiz": "/api/generate/quiz",
            "study_guide": "/api/generate/study-guide",
            "export": "/api/export",
            "email": "/api/email"
        }
    }

@app.get("/ui")
async def serve_ui():
    """Serve the web UI."""
    return FileResponse(static_dir / "index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    response = {
        "pipeline_manager": True if pipeline_manager else False,
    }
    
    if pipeline is None:
        response.update({
            "status": "degraded",
            "message": "Pipeline not initialized - running in limited mode",
            "components": {},
            "registered_tools": []
        })
        return response
    
    try:
        status = pipeline.get_pipeline_status()
        response.update({
            "status": "healthy",
            "components": status.get('components', {}),
            "registered_tools": status.get('registered_tools', [])
        })
        return response
    except Exception as e:
        logger.error(f"Health check error: {e}")
        response.update({
            "status": "unhealthy",
            "error": str(e)
        })
        return response

def _extract_quiz_params_fallback(message: str) -> tuple:
    """Fallback method to extract quiz parameters from message."""
    import re
    message_lower = message.lower()
    
    # Extract number of questions
    num_questions = 5
    num_match = re.search(r'(\d+)\s*questions?', message_lower)
    if num_match:
        num_questions = int(num_match.group(1))
    
    # Extract difficulty
    difficulty = 'medium'
    if any(word in message_lower for word in ['difficult', 'hard', 'challenging']):
        difficulty = 'hard'
    elif any(word in message_lower for word in ['easy', 'simple', 'basic']):
        difficulty = 'easy'
    
    # Extract topic - remove quiz-related words
    topic = message
    # Remove common patterns
    topic = re.sub(r'\b(create|generate|make|a|an|the|quiz|with|level|questions?|difficult|easy|hard|medium|on|about)\b', '', topic, flags=re.IGNORECASE)
    # Remove numbers
    topic = re.sub(r'\d+', '', topic)
    # Clean up whitespace
    topic = re.sub(r'\s+', ' ', topic).strip()
    
    if not topic or len(topic) < 3:
        topic = "sustainability"
    
    return topic, num_questions, difficulty

def _ensure_pipeline_components():
    """Ensure pipeline components are available, initialize if needed."""
    global pipeline
    
    # Check if pipeline exists and has valid components
    if pipeline is not None:
        # CRITICAL: Check enhanced_agent first (not just agent)
        if hasattr(pipeline, 'enhanced_agent'):
            if pipeline.enhanced_agent is not None:
                logger.info("Enhanced agent available from pipeline")
                return True
            else:
                logger.warning("Pipeline has enhanced_agent attribute but it is None - will initialize")
        # Check agent (backward compatibility) - but DON'T return early, we need enhanced_agent
        if hasattr(pipeline, 'agent'):
            if pipeline.agent is not None:
                logger.info("Legacy agent available from pipeline, but will upgrade to enhanced_agent")
                # Don't return - continue to initialize enhanced_agent
            else:
                logger.warning("Pipeline has agent attribute but it is None - will initialize")
        # Check content_generator
        if hasattr(pipeline, 'content_generator'):
            if pipeline.content_generator is not None:
                logger.debug("Content generator available from pipeline")
                # Don't return True here - we still need to check/enhance agent
            else:
                logger.warning("Pipeline has content_generator attribute but it is None")
    
    # Try to initialize components directly if pipeline failed or components are None
    # OR if pipeline exists but doesn't have enhanced_agent
    needs_enhanced_agent = (pipeline is not None and 
                           (not hasattr(pipeline, 'enhanced_agent') or pipeline.enhanced_agent is None))
    
    if needs_enhanced_agent:
        logger.info("Pipeline exists but enhanced_agent is missing - initializing enhanced_agent...")
    else:
        logger.warning("Pipeline components not available, attempting direct initialization...")
    try:
        # Check for API key - try multiple sources
        api_key = os.getenv("GEMINI_API_KEY")
        
        # Try to get from the pipeline if it exists
        if not api_key and pipeline is not None:
            if hasattr(pipeline, 'llm') and pipeline.llm is not None:
                if hasattr(pipeline.llm, 'api_key') and pipeline.llm.api_key:
                    api_key = pipeline.llm.api_key
                    logger.info("Using API key from existing LLM client in pipeline")
        
        # Try hardcoded key as last resort (for development)
        # This is the key provided by the user
        if not api_key:
            api_key = "AIzaSyDcmWDDD9auO6kyPYvMQbgMZRfZq02ydjo"
            logger.info("Using provided API key (development mode)")
        
        if not api_key:
            logger.error("GEMINI_API_KEY environment variable not set")
            logger.error("Please set GEMINI_API_KEY environment variable before starting the server")
            return False
        
        # Set it in environment for the imports
        os.environ["GEMINI_API_KEY"] = api_key
        
        from partd.llm_client import GeminiLLMClient
        from partd.enhanced_rag import EnhancedRAG
        from partd.task_handlers import ContentGenerator
        
        logger.info("Initializing LLM client...")
        llm = GeminiLLMClient()
        
        # Try to use existing vector_retriever from pipeline if available
        retrieval_pipeline = None
        if pipeline is not None and hasattr(pipeline, 'vector_retriever'):
            if pipeline.vector_retriever is not None:
                retrieval_pipeline = pipeline.vector_retriever
                logger.info("Using existing vector retriever from pipeline")
            else:
                logger.warning("Pipeline has vector_retriever but it is None")
        
        # If no vector retriever, try to initialize one
        if retrieval_pipeline is None:
            try:
                from partc.milvus_retrieval_pipeline import MilvusRetrievalPipeline
                from pathlib import Path
                # Try milvus_lite.db first (more recent), fallback to milvus_data.db
                db_path = project_root / 'partc' / 'milvus_lite.db'
                if not db_path.exists():
                    db_path = project_root / 'partc' / 'milvus_data.db'
                logger.info(f"Using Milvus database: {db_path}")
                logger.info(f"Attempting to initialize Milvus retriever from {db_path}")
                retrieval_pipeline = MilvusRetrievalPipeline(
                    str(db_path),
                    prefer_scientific=True  # Use scibert for environment domain
                )
                if hasattr(retrieval_pipeline, 'milvus_available') and not retrieval_pipeline.milvus_available:
                    logger.warning("Milvus collections not available, will use web search fallback")
                    # Create MockRetrieval for fallback
                    class MockRetrieval:
                        def hierarchical_retrieve(self, query, **kwargs):
                            return {'results': []}
                        milvus_available = False
                    retrieval_pipeline = MockRetrieval()
                else:
                    logger.info("✓ Milvus retriever initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Milvus retriever: {e}, using web search fallback")
                # Create MockRetrieval for fallback
                class MockRetrieval:
                    def hierarchical_retrieve(self, query, **kwargs):
                        return {'results': []}
                    milvus_available = False
                retrieval_pipeline = MockRetrieval()
        
        logger.info("Initializing RAG with retrieval pipeline...")
        rag = EnhancedRAG(retrieval_pipeline, llm, enable_web_search=True)
        logger.info("Initializing content generator...")
        content_generator = ContentGenerator(rag, llm)
        
        # Initialize knowledge tools for library retrieval
        knowledge_tools = None
        try:
            from parth.knowledge_tools import KnowledgeTools
            # Try to get hybrid retriever from pipeline if available
            hybrid_retriever = None
            if pipeline is not None and hasattr(pipeline, 'hybrid_retriever'):
                hybrid_retriever = pipeline.hybrid_retriever
            knowledge_tools = KnowledgeTools(rag_system=rag, hybrid_retriever=hybrid_retriever, llm_client=llm)
            logger.info("✓ Knowledge tools initialized")
        except Exception as e:
            logger.warning(f"Could not initialize knowledge tools: {e}")
        
        # Initialize enhanced multi-agent system
        enhanced_agent = None
        try:
            logger.info("Attempting to import EnhancedMultiAgent...")
            from parte.enhanced_multi_agent import EnhancedMultiAgent
            logger.info("✓ EnhancedMultiAgent imported successfully")
            
            logger.info("Initializing EnhancedMultiAgent with components...")
            enhanced_agent = EnhancedMultiAgent(
                llm_client=llm,
                rag_system=rag,
                content_generator=content_generator,
                knowledge_tools=knowledge_tools
            )
            logger.info("✓ Enhanced multi-agent system initialized successfully")
            logger.info(f"✓ EnhancedMultiAgent object created: {type(enhanced_agent).__name__}")
        except ImportError as e:
            logger.error(f"✗ Failed to import EnhancedMultiAgent: {e}")
            import traceback
            logger.error(traceback.format_exc())
            enhanced_agent = None
        except Exception as e:
            logger.error(f"✗ Could not initialize enhanced multi-agent: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Set to None but log the error clearly
            enhanced_agent = None
        
        # Update existing pipeline or create new one
        if pipeline is not None:
            # Update existing pipeline with new components
            pipeline.content_generator = content_generator
            pipeline.llm = llm
            pipeline.rag = rag
            # CRITICAL: Always set enhanced_agent, even if it's None (so we know it was attempted)
            pipeline.enhanced_agent = enhanced_agent
            # CRITICAL: REPLACE legacy agent with enhanced_agent (or None if initialization failed)
            # This ensures we don't accidentally use the old agent_core
            if enhanced_agent is not None:
                pipeline.agent = enhanced_agent  # Replace legacy agent with enhanced_agent
                logger.info("✓ Replaced legacy agent with enhanced_agent")
            else:
                # If enhanced_agent is None, set agent to None too to prevent fallback
                logger.warning("⚠ Setting pipeline.agent to None (enhanced_agent initialization failed)")
                pipeline.agent = None  # Prevent fallback to broken legacy agent
            pipeline.knowledge_tools = knowledge_tools
            # Logging already done above when setting agent
            
            # Update vector_retriever if we successfully initialized one (not MockRetrieval)
            if retrieval_pipeline is not None:
                # Check if it's a real retriever (has milvus_available and it's True, or doesn't have the attribute)
                is_mock = (hasattr(retrieval_pipeline, 'milvus_available') and 
                          retrieval_pipeline.milvus_available == False)
                if not is_mock:
                    pipeline.vector_retriever = retrieval_pipeline
                    logger.info("✓ Updated pipeline with vector retriever")
            
            # Update knowledge_tools if it exists
            if hasattr(pipeline, 'knowledge_tools') and pipeline.knowledge_tools is not None:
                pipeline.knowledge_tools.rag = rag
                pipeline.knowledge_tools.llm = llm
                logger.info("✓ Updated knowledge_tools with new RAG instance")
            
            logger.info("✓ Updated existing pipeline with new components")
        else:
            # Create a minimal pipeline object
            class MinimalPipeline:
                def __init__(self, llm, rag, content_generator, enhanced_agent=None, knowledge_tools=None):
                    self.llm = llm
                    self.rag = rag
                    self.content_generator = content_generator
                    self.enhanced_agent = enhanced_agent
                    self.agent = enhanced_agent  # Alias for backward compatibility
                    self.knowledge_tools = knowledge_tools
                    self.hybrid_retriever = None
                    self.vector_retriever = None
            
            pipeline = MinimalPipeline(llm, rag, content_generator, enhanced_agent, knowledge_tools)
            logger.info("✓ Created new minimal pipeline with components")
            if enhanced_agent is not None:
                logger.info("✓ Minimal pipeline created with enhanced_agent")
            else:
                logger.error("✗ Minimal pipeline created but enhanced_agent is None")
        
        # CRITICAL: Verify enhanced_agent is set before returning
        if pipeline is not None:
            if hasattr(pipeline, 'enhanced_agent') and pipeline.enhanced_agent is not None:
                logger.info("✓ Verified: enhanced_agent is set on pipeline")
                logger.info(f"✓ enhanced_agent type: {type(pipeline.enhanced_agent).__name__}")
                return True
            else:
                logger.error("✗ ERROR: enhanced_agent was not set on pipeline after initialization!")
                logger.error(f"  enhanced_agent variable is: {enhanced_agent}")
                logger.error(f"  pipeline.enhanced_agent is: {getattr(pipeline, 'enhanced_agent', 'NOT SET')}")
                # Try one more time to set it
                if enhanced_agent is not None:
                    pipeline.enhanced_agent = enhanced_agent
                    pipeline.agent = enhanced_agent
                    logger.info("✓ Manually set enhanced_agent on pipeline")
                    logger.info(f"✓ After manual set, pipeline.enhanced_agent is: {pipeline.enhanced_agent is not None}")
                    return True
                else:
                    logger.error("✗ Cannot set enhanced_agent - it is None (initialization failed)")
        
        return True
    except Exception as e:
        logger.error(f"Failed to initialize components directly: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Main chat endpoint - uses Enhanced Multi-Agent System for intelligent processing."""
    # Apply input guardrails
    try:
        from parti.guardrails import apply_input_guardrails
        sanitized_message, validation = apply_input_guardrails(request.message)
        if validation.get("warnings"):
            logger.warning(f"Input guardrail warnings: {validation['warnings']}")
        # Use sanitized input
        request.message = sanitized_message
    except Exception as e:
        logger.warning(f"Input guardrails error (non-blocking): {e}")
        # Continue with original input if guardrails fail
    
    # Ensure components are available
    if not _ensure_pipeline_components():
        raise HTTPException(status_code=503, detail="Failed to initialize pipeline components")
    
    try:
        # Try using enhanced multi-agent first
        has_enhanced_agent = hasattr(pipeline, 'enhanced_agent') and pipeline.enhanced_agent is not None
        logger.info(f"Enhanced agent available: {has_enhanced_agent}")
        if has_enhanced_agent:
            logger.info(f"Processing request with EnhancedMultiAgent: {request.message[:100]}")
            result = pipeline.enhanced_agent.process_request(request.message)
            
            # Determine response type - prioritize content type over task type for UI display
            response_type = result.get('type', 'answer')  # Default to 'answer' for Q&A
            if result.get('quiz'):
                response_type = 'quiz'
            elif result.get('study_guide'):
                response_type = 'study_guide'
            elif result.get('awareness_content'):
                response_type = 'awareness'
            elif result.get('type') == 'awareness_email':
                response_type = 'awareness_email'
            elif result.get('type') == 'library_upload':
                response_type = 'library_upload'
            elif result.get('type') == 'answer':
                response_type = 'answer'  # Explicitly set for Q&A responses
            
            # Apply output guardrails to response
            response_text = result.get('message', result.get('response', ''))
            try:
                from parti.guardrails import apply_output_guardrails
                moderated_response, moderation = apply_output_guardrails(response_text, content_type=response_type)
                if moderation.get("warnings"):
                    logger.warning(f"Output guardrail warnings: {moderation['warnings']}")
                if moderation.get("blocked"):
                    logger.error(f"Output blocked due to harmful content")
                response_text = moderated_response
            except Exception as e:
                logger.warning(f"Output guardrails error (non-blocking): {e}")
                # Continue with original response if guardrails fail
            
            # For awareness_email, don't show awareness_content, show email status
            response_data = {
                "success": result.get('success', True),
                "response": response_text,
                "type": response_type,
                "quiz": result.get('quiz'),
                "study_guide": result.get('study_guide'),
                "topic": result.get('quiz', {}).get('topic') or result.get('study_guide', {}).get('topic') or result.get('awareness_content', {}).get('topic'),
                "metadata": result.get('metadata', {}),
                "sources": result.get('sources')  # Include sources for Q&A responses
            }
            
            # Include show_upload_ui flag ONLY if explicitly set to True (not just truthy)
            if result.get('show_upload_ui') is True:
                response_data["show_upload_ui"] = True
            
            # Only include awareness_content if it's not an awareness_email type
            if response_type != 'awareness_email':
                response_data["awareness_content"] = result.get('awareness_content')
            else:
                # For awareness_email, include email metadata
                response_data["email_metadata"] = result.get('metadata', {})
            
            return response_data
        
        # CRITICAL: Don't fallback to old agent - it doesn't handle awareness_email correctly
        # If enhanced_agent is not available, we should have initialized it above
        # Only use old agent as absolute last resort, and log a warning
        if hasattr(pipeline, 'agent') and pipeline.agent is not None:
            # Check if this is actually the enhanced_agent (shouldn't happen if enhanced_agent check above worked)
            agent_type = type(pipeline.agent).__name__
            if agent_type == 'EnhancedMultiAgent':
                logger.warning("Using pipeline.agent which is EnhancedMultiAgent (should have been caught above)")
                result = pipeline.agent.process_request(request.message)
            else:
                logger.error(f"✗ Enhanced agent not available, falling back to legacy agent ({agent_type})")
                logger.error("✗ Legacy agent does not support awareness_email - request will fail")
                logger.error("✗ Please check why enhanced_agent initialization failed in logs above")
                # Try the legacy agent anyway, but it won't work for awareness_email
                result = pipeline.agent.process_request(request.message)
            # Ensure sources are included for Q&A responses
            response_data = {
                "success": True,
                "response": result.get('message', result.get('response', '')),
                "type": result.get('type', 'answer'),
                "metadata": result.get('metadata', {}),
                "sources": result.get('sources', {})  # Include sources for Q&A responses
            }
            
            # Include quiz/study_guide data if present (for content_generator handler)
            if result.get('quiz'):
                response_data['quiz'] = result.get('quiz')
                response_data['type'] = 'quiz'
                response_data['topic'] = result.get('topic') or result.get('quiz', {}).get('topic', '')
            elif result.get('study_guide'):
                response_data['study_guide'] = result.get('study_guide')
                response_data['type'] = 'study_guide'
                response_data['topic'] = result.get('topic') or result.get('study_guide', {}).get('topic', '')
            elif result.get('content'):
                # Check if content is a quiz or study guide
                content = result.get('content', {})
                if content.get('content_type') == 'quiz':
                    response_data['quiz'] = content
                    response_data['type'] = 'quiz'
                    response_data['topic'] = content.get('topic', '')
                elif content.get('content_type') == 'study_guide':
                    response_data['study_guide'] = content
                    response_data['type'] = 'study_guide'
                    response_data['topic'] = content.get('topic', '')
            
            # Apply output guardrails to response text if present
            if response_data.get('response'):
                try:
                    from parti.guardrails import apply_output_guardrails
                    moderated_response, moderation = apply_output_guardrails(response_data['response'], content_type=response_data.get('type', 'text'))
                    if moderation.get("warnings"):
                        logger.warning(f"Output guardrail warnings: {moderation['warnings']}")
                    if moderation.get("blocked"):
                        logger.error(f"Output blocked due to harmful content")
                    response_data['response'] = moderated_response
                except Exception as e:
                    logger.warning(f"Output guardrails error (non-blocking): {e}")
                    # Continue with original response if guardrails fail
            
            return response_data
        
        # Fallback: Use content generator directly for Q&A
        if hasattr(pipeline, 'content_generator') and pipeline.content_generator is not None:
            logger.info("Agent not available, using content generator directly")
            # Check if it's a question
            message_lower = request.message.lower()
            if any(word in message_lower for word in ['what', 'how', 'why', 'when', 'where', 'explain', 'tell me', '?']):
                result = pipeline.content_generator.answer_question(request.message)
                # Extract sources from result
                sources = result.get('sources', {})
                if not sources:
                    # Try to get sources from metadata or construct from result
                    metadata = result.get('metadata', {})
                    sources = {
                        "local": result.get('local_sources', []) or [],
                        "web": result.get('web_sources', []) or []
                    }
                
                # Apply output guardrails to answer
                answer = result.get('answer', '')
                try:
                    from parti.guardrails import apply_output_guardrails
                    moderated_answer, moderation = apply_output_guardrails(answer, content_type="answer")
                    if moderation.get("warnings"):
                        logger.warning(f"Output guardrail warnings: {moderation['warnings']}")
                    if moderation.get("blocked"):
                        logger.error(f"Output blocked due to harmful content")
                    answer = moderated_answer
                except Exception as e:
                    logger.warning(f"Output guardrails error (non-blocking): {e}")
                    # Continue with original answer if guardrails fail
                
                return {
                    "success": True,
                    "response": answer,
                    "type": "answer",
                    "metadata": result.get('metadata', {}),
                    "sources": sources  # Always include sources
                }
            # Check if it's a quiz request
            elif 'quiz' in message_lower:
                # Extract parameters from message using LLM for better parsing
                try:
                    extraction_prompt = f"""Extract quiz generation parameters from this user request:

"{request.message}"

Extract:
1. Topic/subject (what the quiz should be about)
2. Number of questions (if specified, default is 5)
3. Difficulty level (easy/medium/hard/difficult, default is medium)

Return JSON:
{{
  "topic": "clear topic name",
  "num_questions": number,
  "difficulty": "easy|medium|hard"
}}

Be precise. If topic mentions "animals and environment", use "animals and environment education" as topic.
If difficulty says "difficult", use "hard". If it says "difficult level", still extract the actual topic."""

                    extraction_result = pipeline.llm.generate_json(extraction_prompt)
                    
                    if extraction_result:
                        topic = extraction_result.get('topic', '').strip()
                        num_questions = extraction_result.get('num_questions', 5)
                        difficulty = extraction_result.get('difficulty', 'medium').lower()
                        
                        # Validate and normalize
                        if not topic or len(topic) < 3:
                            # Fallback: try simple extraction
                            import re
                            # Remove common quiz-related words
                            topic = re.sub(r'\b(quiz|generate|create|make|about|on|with|level|questions?|difficult|easy|hard|medium)\b', '', message_lower, flags=re.IGNORECASE)
                            topic = re.sub(r'\s+', ' ', topic).strip()
                            if not topic or len(topic) < 3:
                                topic = "sustainability"
                        
                        # Normalize difficulty
                        if difficulty in ['difficult', 'difficulty']:
                            difficulty = 'hard'
                        elif difficulty not in ['easy', 'medium', 'hard']:
                            difficulty = 'medium'
                        
                        # Ensure num_questions is valid
                        try:
                            num_questions = int(num_questions)
                            if num_questions < 1 or num_questions > 20:
                                num_questions = 5
                        except:
                            num_questions = 5
                        
                        logger.info(f"Extracted quiz params: topic='{topic}', num_questions={num_questions}, difficulty={difficulty}")
                    else:
                        # Fallback extraction
                        topic, num_questions, difficulty = _extract_quiz_params_fallback(request.message)
                except Exception as e:
                    logger.warning(f"LLM extraction failed: {e}, using fallback")
                    topic, num_questions, difficulty = _extract_quiz_params_fallback(request.message)
                
                quiz = pipeline.content_generator.generate_quiz(topic, num_questions=num_questions, difficulty=difficulty)
                
                # Format quiz for display - structured format
                # Use the topic from quiz response, fallback to extracted topic
                quiz_topic = quiz.get('topic', topic)
                if not quiz_topic or len(quiz_topic.strip()) < 3:
                    quiz_topic = topic
                quiz_topic = quiz_topic.strip()
                
                quiz_text = f"📝 Quiz: {quiz_topic.title()}\n"
                quiz_text += f"Difficulty: {quiz.get('difficulty', difficulty).title()} | Questions: {len(quiz.get('questions', []))}\n"
                quiz_text += "=" * 60 + "\n\n"
                
                for i, q in enumerate(quiz.get('questions', []), 1):
                    quiz_text += f"{i}. {q.get('question', '')}\n\n"
                    # MCQ options
                    for key in ['A', 'B', 'C', 'D']:
                        if key in q.get('options', {}):
                            quiz_text += f"   {key}) {q['options'][key]}\n"
                    quiz_text += "\n"
                    quiz_text += f"   Answer: {q.get('correct_answer', '')}\n"
                    quiz_text += f"   Explanation: {q.get('explanation', '')}\n"
                    quiz_text += "\n" + "-" * 60 + "\n\n"
                
                return {
                    "success": True,
                    "response": quiz_text,
                    "type": "quiz",
                    "quiz": quiz,
                    "topic": topic
                }
            # Check if it's a study guide request
            elif 'study guide' in message_lower or 'study-guide' in message_lower or 'studyguide' in message_lower:
                # Try to extract topic
                topic = request.message.replace('study guide', '').replace('study-guide', '').replace('studyguide', '').replace('generate', '').replace('create', '').replace('a', '').replace('on', '').strip()
                if not topic or len(topic) < 3:
                    topic = "sustainability"
                study_guide = pipeline.content_generator.generate_study_guide(topic=topic)
                
                # Format study guide for display
                guide_text = f"📚 Study Guide: {study_guide.get('topic', topic).title()}\n"
                guide_text += "=" * 60 + "\n\n"
                guide_text += study_guide.get('content', '')
                
                return {
                    "success": True,
                    "response": guide_text,
                    "type": "study_guide",
                    "study_guide": study_guide,
                    "topic": topic
                }
            else:
                # Generic response - treat as question
                result = pipeline.content_generator.answer_question(request.message)
                return {
                    "success": True,
                    "response": result.get('answer', ''),
                    "type": "answer",
                    "metadata": result.get('metadata', {})
                }
        
        raise HTTPException(status_code=503, detail="Agent and content generator not available")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query_knowledge(request: QueryRequest):
    """Knowledge retrieval endpoint."""
    # Apply input guardrails
    try:
        from parti.guardrails import apply_input_guardrails
        sanitized_query, validation = apply_input_guardrails(request.query)
        if validation.get("warnings"):
            logger.warning(f"Input guardrail warnings: {validation['warnings']}")
        # Use sanitized input
        request.query = sanitized_query
    except Exception as e:
        logger.warning(f"Input guardrails error (non-blocking): {e}")
        # Continue with original input if guardrails fail
    global pipeline
    
    # Ensure pipeline components are available
    if not _ensure_pipeline_components():
        raise HTTPException(status_code=503, detail="Failed to initialize pipeline components")
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Ensure knowledge_tools exists and is updated with latest RAG/hybrid
    if not hasattr(pipeline, 'knowledge_tools') or pipeline.knowledge_tools is None:
        logger.info("Knowledge tools not found, creating new instance...")
        try:
            from parth.knowledge_tools import KnowledgeTools
            
            rag = getattr(pipeline, 'rag', None)
            hybrid = getattr(pipeline, 'hybrid_retriever', None)
            llm = getattr(pipeline, 'llm', None)
            
            pipeline.knowledge_tools = KnowledgeTools(rag, hybrid, llm)
            logger.info("✓ Knowledge tools created")
        except Exception as e:
            logger.error(f"Failed to create knowledge tools: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=503, detail=f"Knowledge tools not available: {str(e)}")
    else:
        # Update knowledge_tools with latest RAG/hybrid if they were reinitialized
        rag = getattr(pipeline, 'rag', None)
        hybrid = getattr(pipeline, 'hybrid_retriever', None)
        llm = getattr(pipeline, 'llm', None)
        
        if pipeline.knowledge_tools.rag != rag or pipeline.knowledge_tools.hybrid != hybrid:
            logger.info("Updating knowledge tools with latest RAG/hybrid instances...")
            pipeline.knowledge_tools.rag = rag
            pipeline.knowledge_tools.hybrid = hybrid
            pipeline.knowledge_tools.llm = llm
    
    try:
        logger.info(f"Query request: '{request.query}' | Method: {request.method} | Top-K: {request.top_k}")
        
        # Log available components
        has_rag = hasattr(pipeline, 'rag') and pipeline.rag is not None
        has_hybrid = hasattr(pipeline, 'hybrid_retriever') and pipeline.hybrid_retriever is not None
        has_vector = hasattr(pipeline, 'vector_retriever') and pipeline.vector_retriever is not None
        
        logger.info(f"Available components - RAG: {has_rag}, Hybrid: {has_hybrid}, Vector: {has_vector}")
        
        result = pipeline.knowledge_tools.retrieve_knowledge(
            query=request.query,
            method=request.method,
            top_k=request.top_k,
            similarity_cutoff=request.similarity_cutoff
        )
        
        # Log result summary
        if result.get('success'):
            logger.info(f"✓ Query successful: {result.get('num_results', 0)} results")
            if result.get('metadata'):
                metadata = result['metadata']
                if 'used_web_search' in metadata:
                    logger.info(f"  Library usage: {'Web search fallback' if metadata['used_web_search'] else 'Local library'}")
                if 'retrieval_confidence' in metadata:
                    logger.info(f"  Confidence: {metadata['retrieval_confidence']:.3f}")
        else:
            logger.warning(f"✗ Query failed: {result.get('error', 'Unknown error')}")
        
        # Apply output guardrails to answer if present
        if result.get('answer'):
            try:
                from parti.guardrails import apply_output_guardrails
                moderated_answer, moderation = apply_output_guardrails(result['answer'], content_type="text")
                if moderation.get("warnings"):
                    logger.warning(f"Output guardrail warnings: {moderation['warnings']}")
                if moderation.get("blocked"):
                    logger.error(f"Output blocked due to harmful content")
                result['answer'] = moderated_answer
            except Exception as e:
                logger.warning(f"Output guardrails error (non-blocking): {e}")
                # Continue with original answer if guardrails fail
        
        return result
    
    except Exception as e:
        logger.error(f"Query error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate/quiz")
async def generate_quiz(request: QuizRequest):
    """Generate quiz endpoint."""
    # Ensure components are available
    if not _ensure_pipeline_components():
        raise HTTPException(status_code=503, detail="Failed to initialize pipeline components")
    
    if not hasattr(pipeline, 'content_generator') or pipeline.content_generator is None:
        raise HTTPException(status_code=503, detail="Content generator not available")
    
    try:
        quiz = pipeline.content_generator.generate_quiz(
            topic=request.topic,
            num_questions=request.num_questions,
            difficulty=request.difficulty
        )
        
        return {
            "success": True,
            "quiz": quiz,
            "message": f"Quiz generated successfully with {len(quiz.get('questions', []))} questions on {request.topic}"
        }
    
    except Exception as e:
        logger.error(f"Quiz generation error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate/study-guide")
async def generate_study_guide(request: StudyGuideRequest):
    """Generate study guide endpoint."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if not hasattr(pipeline, 'content_generator') or pipeline.content_generator is None:
        raise HTTPException(status_code=503, detail="Content generator not available")
    
    try:
        guide = pipeline.content_generator.generate_study_guide(topic=request.topic)
        
        return {
            "success": True,
            "study_guide": guide
        }
    
    except Exception as e:
        logger.error(f"Study guide generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/export")
async def export_document(request: DocumentExportRequest):
    """Document export endpoint."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if not hasattr(pipeline, 'document_tools') or pipeline.document_tools is None:
        raise HTTPException(status_code=503, detail="Document tools not available")
    
    try:
        result = pipeline.document_tools.generate_document(
            content=request.content,
            content_type=request.content_type,
            preferred_format=request.format
        )
        
        if result.get('success'):
            filepath = result.get('filepath')
            if filepath and Path(filepath).exists():
                return FileResponse(
                    filepath,
                    media_type='application/octet-stream',
                    filename=Path(filepath).name
                )
            else:
                raise HTTPException(status_code=500, detail="Generated file not found")
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
    
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/email")
async def send_email(request: EmailRequest, background_tasks: BackgroundTasks):
    """Email sending endpoint."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if not hasattr(pipeline, 'email_tools') or pipeline.email_tools is None:
        raise HTTPException(status_code=503, detail="Email tools not available")
    
    try:
        # Send email in background
        background_tasks.add_task(
            pipeline.email_tools.send_with_retry,
            recipient=request.recipient,
            subject=request.subject,
            body=request.body,
            attachments=request.attachments
        )
        
        return {
            "success": True,
            "message": "Email queued for sending"
        }
    
    except Exception as e:
        logger.error(f"Email error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/send-email")
async def send_generated_content_email(request: SendEmailRequest):
    """Send generated content (quiz/study guide) via email with document attachment."""
    # Ensure components are available
    if not _ensure_pipeline_components():
        raise HTTPException(status_code=503, detail="Failed to initialize pipeline components")
    
    try:
        # Get document tools and email tools
        document_tools = None
        email_tools = None
        
        if pipeline is not None:
            if hasattr(pipeline, 'document_tools') and pipeline.document_tools is not None:
                document_tools = pipeline.document_tools
            if hasattr(pipeline, 'email_tools') and pipeline.email_tools is not None:
                email_tools = pipeline.email_tools
        
        # Initialize tools if not available
        if not document_tools:
            from parth.document_tools import DocumentTools
            document_tools = DocumentTools()
        
        if not email_tools:
            from parth.email_tools import EmailTools
            smtp_config = {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'sender_email': os.getenv("SENDER_EMAIL", "sethharry123@gmail.com"),
                'sender_password': os.getenv("SENDER_PASSWORD", "jkex gegg zrhm ryig")
            }
            email_tools = EmailTools(smtp_config)
        
        # Generate document
        content_type = 'quiz' if request.content_type == 'quiz' else 'study_guide'
        content_data = request.content.get('quiz') if request.content_type == 'quiz' else request.content.get('study_guide', request.content)
        
        logger.info(f"Generating {request.format} document for {request.topic}...")
        doc_result = document_tools.generate_document(
            content=content_data,
            content_type=content_type,
            preferred_format=request.format
        )
        
        if not doc_result.get('success'):
            raise HTTPException(status_code=500, detail=f"Failed to generate document: {doc_result.get('error')}")
        
        filepath = doc_result.get('filepath')
        if not filepath or not Path(filepath).exists():
            raise HTTPException(status_code=500, detail="Generated document file not found")
        
        # Prepare email
        topic_title = request.topic.title()
        if request.content_type == 'quiz':
            subject = f"Environmental Sustainability Quiz: {topic_title}"
            body = f"""Hello,

I'm sharing an educational quiz on {topic_title} that was generated by our Environmental Sustainability Assistant.

This quiz contains {len(content_data.get('questions', []))} questions at {content_data.get('difficulty', 'medium')} difficulty level, designed to help you learn more about environmental sustainability.

The quiz document is attached in {request.format.upper()} format. You can use it for educational purposes, awareness programs, or personal learning.

Best regards,
Environmental Sustainability Assistant"""
        else:
            subject = f"Environmental Sustainability Study Guide: {topic_title}"
            body = f"""Hello,

I'm sharing a comprehensive study guide on {topic_title} that was generated by our Environmental Sustainability Assistant.

This study guide contains detailed information and insights about {topic_title}, designed to help you understand key concepts and practices related to environmental sustainability.

The study guide document is attached in {request.format.upper()} format. You can use it for educational purposes, awareness programs, or personal learning.

Best regards,
Environmental Sustainability Assistant"""
        
        # Send email with attachment
        logger.info(f"Sending email to {request.recipient} with attachment: {filepath}")
        email_result = email_tools.send_with_retry(
            recipient=request.recipient,
            subject=subject,
            body=body,
            attachments=[filepath],
            max_retries=3
        )
        
        if email_result.get('success'):
            return {
                "success": True,
                "message": f"Email sent successfully to {request.recipient}",
                "document_format": request.format,
                "document_path": filepath
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to send email: {email_result.get('error')}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Send email error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/awareness-email/batch")
async def send_awareness_emails_batch(request: AwarenessEmailRequest):
    """Generate and send awareness emails to multiple recipients."""
    # Ensure components are available
    if not _ensure_pipeline_components():
        raise HTTPException(status_code=503, detail="Failed to initialize pipeline components")
    
    try:
        # Get email tools
        email_tools = None
        email_tool = None
        
        if pipeline is not None:
            if hasattr(pipeline, 'email_tools') and pipeline.email_tools is not None:
                email_tools = pipeline.email_tools
            if hasattr(pipeline, 'llm') and pipeline.llm is not None:
                # Initialize EmailTool for generating email content
                from partd.email_tool import EmailTool
                smtp_config = {
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'sender_email': os.getenv("SENDER_EMAIL", "sethharry123@gmail.com"),
                    'sender_password': os.getenv("SENDER_PASSWORD", "jkex gegg zrhm ryig")
                }
                email_tool = EmailTool(
                    smtp_server=smtp_config['smtp_server'],
                    smtp_port=smtp_config['smtp_port'],
                    sender_email=smtp_config['sender_email'],
                    sender_password=smtp_config['sender_password'],
                    llm_client=pipeline.llm
                )
        
        # Initialize if not available
        if not email_tools:
            from parth.email_tools import EmailTools
            smtp_config = {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'sender_email': os.getenv("SENDER_EMAIL", "sethharry123@gmail.com"),
                'sender_password': os.getenv("SENDER_PASSWORD", "jkex gegg zrhm ryig")
            }
            email_tools = EmailTools(smtp_config)
        
        if not email_tool:
            from partd.email_tool import EmailTool
            email_tool = EmailTool(
                smtp_server='smtp.gmail.com',
                smtp_port=587,
                sender_email=os.getenv("SENDER_EMAIL", "sethharry123@gmail.com"),
                sender_password=os.getenv("SENDER_PASSWORD", "jkex gegg zrhm ryig"),
                llm_client=pipeline.llm if pipeline and hasattr(pipeline, 'llm') and pipeline.llm else None
            )
        
        logger.info(f"Generating awareness email for topic: {request.topic}, audience: {request.audience}")
        
        # Generate email content using EmailTool
        email_body = email_tool.generate_outreach_email_with_reasoning(
            topic=request.topic,
            audience=request.audience
        )
        
        # Generate subject line
        subject = f"Environmental Awareness: {request.topic.title()}"
        
        logger.info(f"Generated email content. Sending to {len(request.recipients)} recipients...")
        
        # Send batch emails
        batch_result = email_tools.send_batch(
            recipients=request.recipients,
            subject=subject,
            body=email_body,
            max_retries=2
        )
        
        # Prepare response
        successful = batch_result.get('successful', 0)
        failed = batch_result.get('failed', 0)
        results = batch_result.get('results', [])
        
        return {
            "success": True,
            "subject": subject,
            "email_body": email_body,
            "total_recipients": len(request.recipients),
            "successful": successful,
            "failed": failed,
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Awareness email batch error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/library-status")
async def get_library_status():
    """Get library status including document counts."""
    try:
        data_json_dir = project_root / "data_json"
        data_for_finetuning_dir = project_root / "data_for_finetuning"
        chunked_dir = project_root / "partb" / "chunked_data"
        cleaned_dir = project_root / "partb" / "cleaned_data"
        
        # Count files
        json_files = list(data_json_dir.glob("*.json")) if data_json_dir.exists() else []
        pending_files = list(data_for_finetuning_dir.glob("*")) if data_for_finetuning_dir.exists() else []
        chunked_files = list(chunked_dir.glob("*_chunked.json")) if chunked_dir.exists() else []
        cleaned_files = list(cleaned_dir.glob("*_cleaned.json")) if cleaned_dir.exists() else []
        
        # Get recent documents (last 10)
        recent_docs = sorted(json_files, key=lambda p: p.stat().st_mtime, reverse=True)[:10]
        
        # Try to get indexing info
        indexing_summary = {}
        indexing_file = project_root / "partc" / "indexing_summary.json"
        if indexing_file.exists():
            try:
                import json
                with open(indexing_file, 'r') as f:
                    indexing_summary = json.load(f)
            except:
                pass
        
        return {
            "library": {
                "total_documents": len(json_files),
                "pending_processing": len(pending_files),
                "chunked_documents": len(chunked_files),
                "cleaned_documents": len(cleaned_files),
                "recent_documents": [f.name for f in recent_docs]
            },
            "indexing": indexing_summary.get("indexing_summary", {}),
            "status": "ready"
        }
    except Exception as e:
        logger.error(f"Library status error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/statistics")
async def get_statistics():
    """Get system statistics."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        status = pipeline.get_pipeline_status()
        return {
            "components": status.get('components', {}),
            "statistics": status.get('statistics', {})
        }
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/execute")
async def execute_workflow(request: Dict[str, Any]):
    """Execute custom workflow."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        result = pipeline.execute_complete_workflow(request.get('user_request', ''))
        return result
    
    except Exception as e:
        logger.error(f"Workflow error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    auto_index: str = Form("true")
):
    """
    Upload and process a document.
    
    Steps:
    1. Save file to data_for_finetuning/
    2. Process using Part B pipeline (extract, clean, chunk)
    3. Convert to JSON and save to data_json/
    4. Optionally rebuild index (Part C)
    """
    auto_index_bool = auto_index.lower() in ('true', '1', 'yes', 'on')
    try:
        # Ensure directories exist
        data_for_finetuning_dir = project_root / "data_for_finetuning"
        data_json_dir = project_root / "data_json"
        data_for_finetuning_dir.mkdir(exist_ok=True, parents=True)
        data_json_dir.mkdir(exist_ok=True, parents=True)
        
        # Save uploaded file
        file_path = data_for_finetuning_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"File uploaded: {file.filename} ({len(content)} bytes)")
        
        # Process file using Part B pipeline
        try:
            sys.path.insert(0, str(project_root / "partb"))
            from batch_pipeline import BatchIngestionPipeline
            
            # Initialize pipeline
            output_base_dir = project_root / "partb" / "output"
            pipeline = BatchIngestionPipeline(
                dataset_dir=str(project_root / "dataset"),
                data_for_finetuning_dir=str(data_for_finetuning_dir),
                data_json_dir=str(data_json_dir),
                output_base_dir=str(output_base_dir)
            )
            
            # Process the file
            logger.info(f"Processing file: {file_path}")
            result = pipeline.process_new_file(file_path)
            
            if not result.get("success", False):
                return {
                    "success": False,
                    "error": result.get("error", "Processing failed"),
                    "stages": result.get("stages", {})
                }
            
            # Get JSON file path
            json_filename = file_path.stem + '.json'
            json_path = data_json_dir / json_filename
            
            if not json_path.exists():
                return {
                    "success": False,
                    "error": "JSON file was not created",
                    "stages": result.get("stages", {})
                }
            
            # Process and index the new document
            index_result = None
            if auto_index_bool:
                logger.info("="*60)
                logger.info("AUTO-INDEXING: Processing and indexing new document...")
                logger.info("="*60)
                try:
                    import time
                    sys.path.insert(0, str(project_root / "parti"))
                    from document_indexer import process_and_index_document

                    provided_chunked_file = None
                    if result.get("chunked_file_path"):
                        provided_chunked_file = Path(result["chunked_file_path"])
                        logger.info(f"Chunked file reported by pipeline: {provided_chunked_file}")
                        wait_time = 0
                        while not provided_chunked_file.exists() and wait_time < 60:
                            time.sleep(1)
                            wait_time += 1
                        if not provided_chunked_file.exists():
                            logger.warning(f"Chunked file still not available after waiting: {provided_chunked_file}")
                        else:
                            logger.info("Chunked file is ready for indexing.")
                    
                    # Fallback search if provided path missing
                    if not provided_chunked_file or not provided_chunked_file.exists():
                        chunked_data_dir = project_root / "partb" / "chunked_data"
                        backup_dir = project_root / "partb" / "output" / "chunked_data"
                        chunked_filename = f"{json_path.stem}_chunked.json"
                        candidate_paths = [
                            chunked_data_dir / chunked_filename,
                            backup_dir / chunked_filename
                        ]
                        for candidate in candidate_paths:
                            logger.info(f"Looking for chunked file: {candidate}")
                            if candidate.exists():
                                provided_chunked_file = candidate
                                break
                    
                    if provided_chunked_file and provided_chunked_file.exists():
                        logger.info(f"✓ Found chunked file: {provided_chunked_file.name}")
                        logger.info("Starting embedding generation and indexing...")
                        # Process: generate embeddings -> index
                        index_result = process_and_index_document(
                            json_path,
                            project_root,
                            chunk_strategy="medium",
                            chunking_method="content_aware",
                            chunked_file_path=provided_chunked_file
                        )
                        
                        if index_result.get('success', False):
                            chunks_indexed = index_result.get('chunks_indexed', 0)
                            logger.info("="*60)
                            logger.info(f"✓✓✓ Document indexed successfully: {chunks_indexed} chunks")
                            logger.info("="*60)
                            
                            # Reload retrieval pipeline to pick up new data
                            try:
                                from partc.milvus_retrieval_pipeline import MilvusRetrievalPipeline
                                # Use milvus_lite.db (same as used for indexing) for consistency
                                db_path = project_root / 'partc' / 'milvus_lite.db'
                                if not db_path.exists():
                                    db_path = project_root / 'partc' / 'milvus_data.db'
                                logger.info(f"Reloading retrieval pipeline from: {db_path}")
                                new_retriever = MilvusRetrievalPipeline(
                                    str(db_path),
                                    prefer_scientific=True
                                )
                                if new_retriever.milvus_available:
                                    # Update global pipeline's retriever (pipeline is already in scope)
                                    if 'pipeline' in globals() and pipeline and hasattr(pipeline, 'vector_retriever'):
                                        pipeline.vector_retriever = new_retriever
                                    # Update RAG if it exists
                                    if 'pipeline' in globals() and pipeline and hasattr(pipeline, 'rag') and pipeline.rag:
                                        pipeline.rag.retrieval = new_retriever
                                    # Update knowledge_tools if it exists
                                    if 'pipeline' in globals() and pipeline and hasattr(pipeline, 'knowledge_tools') and pipeline.knowledge_tools:
                                        # Update RAG in knowledge_tools
                                        if hasattr(pipeline.knowledge_tools, 'rag'):
                                            pipeline.knowledge_tools.rag.retrieval = new_retriever
                                    logger.info("✓ Updated retrieval pipeline with new index")
                                else:
                                    logger.warning("⚠ New retriever not available - Milvus connection failed")
                            except Exception as e:
                                logger.warning(f"Could not update retrieval pipeline: {e}")
                                import traceback
                                logger.warning(traceback.format_exc())
                        else:
                            error_msg = index_result.get('error', 'Unknown error')
                            logger.error("="*60)
                            logger.error(f"✗✗✗ Indexing failed: {error_msg}")
                            logger.error("="*60)
                            import traceback
                            logger.error(traceback.format_exc())
                    
                except ImportError as e:
                    logger.error(f"Failed to import document indexer: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    index_result = {"success": False, "error": f"Import failed: {str(e)}"}
                except Exception as e:
                    logger.error(f"Indexing failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    index_result = {"success": False, "error": str(e)}
            else:
                logger.info("Auto-indexing disabled. Document processed but not indexed.")
                logger.info("To index later, run: python parti/index_all_documents.py")
            
            return {
                "success": True,
                "message": f"Document '{file.filename}' processed successfully",
                "file": file.filename,
                "json_file": json_filename,
                "stages": result.get("stages", {}),
                "index_rebuilt": auto_index_bool,
                "index_result": index_result
            }
            
        except ImportError as e:
            logger.error(f"Failed to import Part B pipeline: {e}")
            return {
                "success": False,
                "error": f"Failed to import processing pipeline: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Processing error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": f"Processing failed: {str(e)}"
            }
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pipeline/action")
async def pipeline_action(request: PipelineActionRequest):
    """Execute data/indexing/analysis pipeline actions (Parts B-G)."""
    if pipeline_manager is None:
        raise HTTPException(status_code=503, detail="Pipeline manager not available")
    
    try:
        result = pipeline_manager.execute_action(request.action, request.payload)
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        logger.error(f"Pipeline action error ({request.action}): {e}")
        raise HTTPException(status_code=500, detail=str(e))

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server()
    
