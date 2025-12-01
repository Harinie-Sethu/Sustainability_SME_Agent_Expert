"""
Pipeline Integrator
Integrates ALL parts (A-G) into a unified system
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from parth.tool_registry import ToolRegistry
from parth.error_handler import ErrorHandler
from parth.tool_orchestrator import ToolOrchestrator
from parth.document_tools import DocumentTools
from parth.email_tools import EmailTools
from parth.knowledge_tools import KnowledgeTools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineIntegrator:
    """
    Master integrator combining all parts:
    
    Part A: Data Collection and Organization
    Part B: Preprocessing and Chunking
    Part C: Embedding and Indexing (Vector DB + Reranking)
    Part D: Core SME Capabilities (RAG + Content Generation)
    Part E: Agent (Planning, Reasoning, Routing)
    Part F: LLM Experimentation (Prompt Engineering)
    Part G: Advanced RAG (Hybrid Retrieval)
    Part H: Tool Integration (This part)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize complete pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or {}
        
        logger.info("="*70)
        logger.info("INITIALIZING COMPLETE PIPELINE - INTEGRATING ALL PARTS")
        logger.info("="*70)
        
        # Initialize core components
        self.registry = ToolRegistry()
        self.error_handler = ErrorHandler()
        self.orchestrator = ToolOrchestrator(self.registry, self.error_handler)
        
        # Initialize Part-specific components
        self._initialize_parts()
        
        # Register all tools
        self._register_all_tools()
        
        logger.info("="*70)
        logger.info("✓ COMPLETE PIPELINE INITIALIZED")
        logger.info("="*70)
    
    def _initialize_parts(self):
        """Initialize components from all parts."""
        
        # Part C: Vector Retrieval
        logger.info("\n[Part C] Initializing Vector Retrieval...")
        try:
            from partc.milvus_retrieval_pipeline import MilvusRetrievalPipeline
            db_path = self.config.get('milvus_db_path', str(Path(__file__).parent.parent /'partc/milvus_data.db'))
            self.vector_retriever = MilvusRetrievalPipeline(db_path)
            if hasattr(self.vector_retriever, 'milvus_available') and not self.vector_retriever.milvus_available:
                logger.warning("  ⚠ Milvus collections not available, using web search fallback")
                self.vector_retriever = None
            else:
                logger.info("  ✓ Vector retrieval initialized")
        except Exception as e:
            logger.warning(f"  ⚠ Vector retrieval not available: {e}")
            self.vector_retriever = None
        
        # Create MockRetrieval for fallback
        class MockRetrieval:
            def hierarchical_retrieve(self, query, **kwargs):
                return {'results': []}
            milvus_available = False
        
        # Part D: LLM and RAG
        logger.info("\n[Part D] Initializing LLM and RAG...")
        try:
            from partd.llm_client import GeminiLLMClient
            from partd.enhanced_rag import EnhancedRAG
            from partd.task_handlers import ContentGenerator
            
            self.llm = GeminiLLMClient()
            logger.info("  ✓ LLM client initialized")
            
            # Use MockRetrieval if vector retriever is not available
            retrieval_pipeline = self.vector_retriever if self.vector_retriever else MockRetrieval()
            
            self.rag = EnhancedRAG(retrieval_pipeline, self.llm, enable_web_search=True)
            self.content_generator = ContentGenerator(self.rag, self.llm)
            logger.info("  ✓ RAG and Content Generator initialized (with web search fallback)")
        except Exception as e:
            logger.error(f"  ✗ LLM/RAG initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.llm = None
            self.rag = None
            self.content_generator = None
        
        # Part E: Agent
        logger.info("\n[Part E] Initializing Agent...")
        try:
            from parte.agent_core import EnvironmentalAgent
            
            if self.llm and self.rag and self.content_generator:
                self.agent = EnvironmentalAgent(self.llm, self.rag, self.content_generator)
                logger.info("  ✓ Agent initialized")
            else:
                self.agent = None
                logger.warning("  ⚠ Agent not available (missing dependencies)")
        except Exception as e:
            logger.error(f"  ✗ Agent initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.agent = None
        
        # Part G: Hybrid Retrieval
        logger.info("\n[Part G] Initializing Hybrid Retrieval...")
        try:
            from partg.bm25_retriever import BM25Retriever
            from partg.hybrid_retriever import HybridRetriever
            
            if self.vector_retriever:
                self.bm25 = BM25Retriever()
                self.hybrid_retriever = HybridRetriever(
                    self.vector_retriever,
                    self.bm25,
                    alpha=0.6
                )
                logger.info("  ✓ Hybrid retrieval initialized")
            else:
                # Use MockRetrieval for hybrid retriever
                class MockVectorRetriever:
                    def hierarchical_retrieve(self, query, **kwargs):
                        return {'results': []}
                    milvus_available = False
                
                mock_vector = MockVectorRetriever()
                self.bm25 = BM25Retriever()
                self.hybrid_retriever = HybridRetriever(
                    mock_vector,
                    self.bm25,
                    alpha=0.6
                )
                logger.info("  ✓ Hybrid retrieval initialized (with fallback)")
        except Exception as e:
            logger.warning(f"  ⚠ Hybrid retrieval not available: {e}")
            self.hybrid_retriever = None
        
        # Part H: Tool Components
        logger.info("\n[Part H] Initializing Tool Components...")
        self.document_tools = DocumentTools()
        self.email_tools = EmailTools(self.config.get('smtp_config'))
        self.knowledge_tools = KnowledgeTools(self.rag, self.hybrid_retriever, self.llm)
        logger.info("  ✓ Tool components initialized")
    
    def _register_all_tools(self):
        """Register all available tools."""
        logger.info("\nRegistering all tools...")
        
        # Knowledge retrieval tools
        if self.knowledge_tools:
            self.registry.register_tool(
                tool_name='retrieve_knowledge',
                tool_function=self.knowledge_tools.retrieve_knowledge,
                category='knowledge',
                description='Retrieve knowledge using hybrid RAG',
                required_params=['query'],
                optional_params=['method', 'top_k', 'expand_query']
            )
            
            self.registry.register_tool(
                tool_name='search_and_summarize',
                tool_function=self.knowledge_tools.search_and_summarize,
                category='knowledge',
                description='Search and generate summary',
                required_params=['query'],
                optional_params=['max_results']
            )
        
        # Content generation tools
        if self.content_generator:
            self.registry.register_tool(
                tool_name='generate_quiz',
                tool_function=self.content_generator.generate_quiz,
                category='generation',
                description='Generate educational quiz',
                required_params=['topic'],
                optional_params=['num_questions', 'difficulty']
            )
            
            self.registry.register_tool(
                tool_name='generate_study_guide',
                tool_function=self.content_generator.generate_study_guide,
                category='generation',
                description='Generate study guide',
                required_params=['topic'],
                optional_params=[]
            )
            
            self.registry.register_tool(
                tool_name='answer_question',
                tool_function=self.content_generator.answer_question,
                category='knowledge',
                description='Answer question using RAG',
                required_params=['question'],
                optional_params=['conversation_history']
            )
        
        # Document export tools
        self.registry.register_tool(
            tool_name='export_document',
            tool_function=self.document_tools.generate_document,
            category='export',
            description='Export document with fallback',
            required_params=['content', 'content_type'],
            optional_params=['preferred_format', 'fallback_formats'],
            fallback_tool='export_document_simple'
        )
        
        # Email tools
        self.registry.register_tool(
            tool_name='send_email',
            tool_function=self.email_tools.send_with_retry,
            category='communication',
            description='Send email with retry',
            required_params=['recipient', 'subject', 'body'],
            optional_params=['attachments', 'max_retries'],
            retry_config={'max_retries': 3, 'backoff': 2}
        )
        
        self.registry.register_tool(
            tool_name='send_batch_email',
            tool_function=self.email_tools.send_batch,
            category='communication',
            description='Send batch emails',
            required_params=['recipients', 'subject', 'body'],
            optional_params=['attachments', 'max_retries']
        )
        
        # Agent tool (if available)
        if self.agent:
            self.registry.register_tool(
                tool_name='process_request',
                tool_function=self.agent.process_request,
                category='analysis',
                description='Process request with agent',
                required_params=['user_request'],
                optional_params=[]
            )
        
        logger.info(f"✓ Registered {len(self.registry.tools)} tools")
    
    def execute_complete_workflow(self, user_request: str) -> Dict[str, Any]:
        """
        Execute complete end-to-end workflow.
        
        Example: "Create a quiz on solar energy and email it to user@example.com"
        
        Args:
            user_request: User's request
            
        Returns:
            Workflow result
        """
        logger.info("\n" + "="*70)
        logger.info("EXECUTING COMPLETE WORKFLOW")
        logger.info("="*70)
        logger.info(f"Request: {user_request}")
        
        # Use agent if available
        if self.agent:
            logger.info("\nUsing Agent for intelligent orchestration...")
            result = self.agent.process_request(user_request)
            return {
                'success': True,
                'method': 'agent',
                'result': result
            }
        
        # Otherwise, manual orchestration
        logger.info("\nUsing manual orchestration...")
        
        # Create workflow
        workflow = self.orchestrator.create_workflow_template(
            name='complete_workflow',
            description='End-to-end content generation and delivery'
        )
        
        # Add steps
        self.orchestrator.add_workflow_step(
            workflow,
            step_name='Generate Content',
            tool_name='generate_quiz',  # Example
            params={'topic': 'solar energy', 'num_questions': 5},
            output_key='quiz_content',
            retry_config={'max_retries': 2, 'backoff': 2}
        )
        
        self.orchestrator.add_workflow_step(
            workflow,
            step_name='Export Document',
            tool_name='export_document',
            params={
                'content': '$quiz_content',
                'content_type': 'quiz',
                'preferred_format': 'pdf'
            },
            output_key='document_path',
            fallback_tool='export_document_simple'
        )
        
        self.orchestrator.add_workflow_step(
            workflow,
            step_name='Send Email',
            tool_name='send_email',
            params={
                'recipient': 'user@example.com',
                'subject': 'Your Solar Energy Quiz',
                'body': 'Please find attached your quiz.',
                'attachments': ['$document_path']
            },
            required=False,  # Optional step
            retry_config={'max_retries': 3, 'backoff': 2}
        )
        
        # Execute workflow
        result = self.orchestrator.execute_workflow(workflow)
        
        return {
            'success': result['success'],
            'method': 'manual_orchestration',
            'result': result
        }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get status of all pipeline components."""
        return {
            'components': {
                'vector_retrieval': self.vector_retriever is not None,
                'llm': self.llm is not None,
                'rag': self.rag is not None,
                'content_generator': self.content_generator is not None,
                'agent': self.agent is not None,
                'hybrid_retrieval': self.hybrid_retriever is not None,
                'document_tools': True,
                'email_tools': self.email_tools.email_available,
                'knowledge_tools': True
            },
            'registered_tools': len(self.registry.tools),
            'tool_categories': self.registry.tool_categories,
            'statistics': {
                'tool_usage': self.registry.get_usage_statistics(),
                'workflows': self.orchestrator.get_workflow_statistics(),
                'errors': self.error_handler.get_error_statistics()
            }
        }
    
    def demonstrate_capabilities(self) -> Dict[str, Any]:
        """Demonstrate all pipeline capabilities."""
        logger.info("\n" + "="*70)
        logger.info("DEMONSTRATING PIPELINE CAPABILITIES")
        logger.info("="*70)
        
        demos = {}
        
        # Demo 1: Knowledge Retrieval
        if self.knowledge_tools:
            logger.info("\n1. Knowledge Retrieval...")
            try:
                result = self.knowledge_tools.retrieve_knowledge(
                    "What is renewable energy?",
                    method="hybrid",
                    top_k=3
                )
                demos['knowledge_retrieval'] = {
                    'success': result['success'],
                    'num_results': result.get('num_results', 0)
                }
            except Exception as e:
                demos['knowledge_retrieval'] = {'error': str(e)}
        
        # Demo 2: Content Generation
        if self.content_generator:
            logger.info("\n2. Content Generation...")
            try:
                quiz = self.content_generator.generate_quiz("recycling", num_questions=2)
                demos['content_generation'] = {
                    'success': True,
                    'content_type': 'quiz',
                    'num_questions': quiz.get('num_questions', 0)
                }
            except Exception as e:
                demos['content_generation'] = {'error': str(e)}
        
        # Demo 3: Document Export
        logger.info("\n3. Document Export...")
        try:
            if 'content_generation' in demos and demos['content_generation'].get('success'):
                doc_result = self.document_tools.generate_document(
                    content=quiz,
                    content_type='quiz',
                    preferred_format='pdf'
                )
                demos['document_export'] = doc_result
        except Exception as e:
            demos['document_export'] = {'error': str(e)}
        
        # Demo 4: Email (without actually sending)
        logger.info("\n4. Email Capability...")
        demos['email'] = {
            'available': self.email_tools.email_available,
            'note': 'Configured but not sending in demo'
        }
        
        # Demo 5: Agent
        if self.agent:
            logger.info("\n5. Agent Processing...")
            try:
                agent_result = self.agent.process_request("What is composting?")
                demos['agent'] = {
                    'success': True,
                    'response_type': agent_result.get('type')
                }
            except Exception as e:
                demos['agent'] = {'error': str(e)}
        
        return demos
