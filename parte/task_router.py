"""
Task Router for Intelligent Task Routing
Routes user requests to appropriate handlers with decision strategies
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, Any, Optional, Callable
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskRouter:
    """
    Intelligent task router with:
    - Pattern matching for task identification
    - Confidence-based routing decisions
    - Handler registration and management
    - Fallback strategies
    """
    
    def __init__(self, llm_client):
        """
        Initialize task router.
        
        Args:
            llm_client: LLM client for intelligent routing
        """
        self.llm = llm_client
        self.handlers: Dict[str, Callable] = {}
        self.routing_stats: Dict[str, int] = {}
        
        logger.info(" Task router initialized")
    
    def register_handler(self, handler_name: str, handler_function: Callable):
        """
        Register a task handler.
        
        Args:
            handler_name: Name of the handler
            handler_function: Function to handle the task
        """
        self.handlers[handler_name] = handler_function
        self.routing_stats[handler_name] = 0
        logger.info(f" Registered handler: {handler_name}")
    
    def route_task(self, user_request: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Route user request to appropriate handler.
        
        Args:
            user_request: User's request
            context: Optional conversation context
            
        Returns:
            Routing decision with handler and parameters
        """
        logger.info(f"Routing request: {user_request[:100]}...")
        
        # First, try rule-based routing for common patterns
        rule_based = self._rule_based_routing(user_request)
        
        if rule_based and rule_based['confidence'] == 'high':
            logger.info(f" Rule-based routing: {rule_based['handler']}")
            return rule_based
        
        # Use LLM for intelligent routing
        llm_routing = self._llm_based_routing(user_request, context)
        
        # Combine rule-based and LLM routing
        final_routing = self._combine_routing_decisions(rule_based, llm_routing)
        
        # Update stats
        handler = final_routing.get('handler', 'unknown')
        self.routing_stats[handler] = self.routing_stats.get(handler, 0) + 1
        
        logger.info(f" Final routing: {handler} (confidence: {final_routing.get('confidence')})")
        
        return final_routing
    
    def _rule_based_routing(self, user_request: str) -> Dict[str, Any]:
        """
        Rule-based routing using pattern matching.
        
        Args:
            user_request: User's request
            
        Returns:
            Routing decision
        """
        request_lower = user_request.lower()
        
        # QA Handler patterns
        qa_patterns = ['what is', 'explain', 'how does', 'why', 'tell me about', 'define']
        if any(pattern in request_lower for pattern in qa_patterns):
            return {
                "handler": "qa_handler",
                "confidence": "high",
                "reasoning": "Question/explanation pattern detected",
                "extracted_parameters": {"question": user_request}
            }
        
        # Content Generator patterns
        content_patterns = [
            ('quiz', 'create', 'generate'),
            ('study guide', 'guide'),
            ('article', 'write'),
            ('social media', 'post', 'tweet')
        ]
        for patterns in content_patterns:
            if any(p in request_lower for p in patterns):
                return {
                    "handler": "content_generator",
                    "confidence": "high",
                    "reasoning": "Content generation pattern detected",
                    "extracted_parameters": self._extract_content_params(user_request)
                }
        
        # Document Handler patterns
        doc_patterns = ['export', 'download', 'save', 'pdf', 'docx', 'document']
        if any(pattern in request_lower for pattern in doc_patterns):
            return {
                "handler": "document_handler",
                "confidence": "high",
                "reasoning": "Document export pattern detected",
                "extracted_parameters": {"format": self._extract_format(user_request)}
            }
        
        # Email Handler patterns
        email_patterns = ['email', 'send', 'share', 'mail']
        if any(pattern in request_lower for pattern in email_patterns):
            return {
                "handler": "email_handler",
                "confidence": "high",
                "reasoning": "Email/sharing pattern detected",
                "extracted_parameters": self._extract_email_params(user_request)
            }
        
        # Data Analyst patterns
        data_patterns = ['compare', 'analyze', 'trend', 'statistics', 'data', 'chart']
        if any(pattern in request_lower for pattern in data_patterns):
            return {
                "handler": "data_analyst",
                "confidence": "medium",
                "reasoning": "Data analysis pattern detected",
                "extracted_parameters": {}
            }
        
        # Conversation Handler patterns (greetings, help)
        conv_patterns = ['hello', 'hi', 'help', 'what can you', 'thank', 'bye']
        if any(pattern in request_lower for pattern in conv_patterns):
            return {
                "handler": "conversation_handler",
                "confidence": "high",
                "reasoning": "Conversational pattern detected",
                "extracted_parameters": {}
            }
        
        # No clear pattern - low confidence
        return {
            "handler": "conversation_handler",
            "confidence": "low",
            "reasoning": "No clear pattern detected, defaulting to conversation",
            "extracted_parameters": {},
            "requires_clarification": True
        }
    
    def _llm_based_routing(self, user_request: str, context: Optional[str]) -> Dict[str, Any]:
        """
        LLM-based intelligent routing.
        
        Args:
            user_request: User's request
            context: Conversation context
            
        Returns:
            Routing decision
        """
        from parte.prompts import ROUTING_PROMPT
        
        prompt = ROUTING_PROMPT.format(user_request=user_request)
        
        try:
            routing_json = self.llm.generate_json(prompt)
            
            if isinstance(routing_json, dict) and 'handler' in routing_json:
                return routing_json
            else:
                logger.warning("LLM routing returned invalid format")
                return {"handler": "conversation_handler", "confidence": "low"}
        
        except Exception as e:
            logger.error(f"LLM routing failed: {e}")
            return {"handler": "conversation_handler", "confidence": "low"}
    
    def _combine_routing_decisions(self, rule_based: Dict, llm_based: Dict) -> Dict[str, Any]:
        """
        Combine rule-based and LLM routing decisions.
        
        Args:
            rule_based: Rule-based routing result
            llm_based: LLM-based routing result
            
        Returns:
            Final routing decision
        """
        # If rule-based has high confidence, use it
        if rule_based.get('confidence') == 'high':
            return rule_based
        
        # If LLM has high confidence and rule-based is low, use LLM
        if llm_based.get('confidence') == 'high' and rule_based.get('confidence') == 'low':
            return llm_based
        
        # If they agree, use either with combined confidence
        if rule_based.get('handler') == llm_based.get('handler'):
            return {
                **llm_based,
                "confidence": "high",
                "reasoning": f"Both methods agree: {llm_based.get('reasoning', '')}"
            }
        
        # Conflict resolution: prefer rule-based for safety
        logger.warning(f"Routing conflict: rule={rule_based.get('handler')}, llm={llm_based.get('handler')}")
        return {
            **rule_based,
            "confidence": "medium",
            "reasoning": f"Rule-based routing with LLM disagreement",
            "alternative_handler": llm_based.get('handler')
        }
    
    def _extract_content_params(self, request: str) -> Dict[str, Any]:
        """Extract parameters for content generation."""
        params = {}
        request_lower = request.lower()
        
        # Detect content type
        if 'quiz' in request_lower:
            params['content_type'] = 'quiz'
        elif 'study guide' in request_lower or 'guide' in request_lower:
            params['content_type'] = 'study_guide'
        elif 'article' in request_lower:
            params['content_type'] = 'article'
        elif 'social media' in request_lower or 'post' in request_lower:
            params['content_type'] = 'social_media'
        
        # Extract topic (simple approach - take words after "about" or "on")
        for keyword in ['about', 'on', 'regarding']:
            if keyword in request_lower:
                parts = request_lower.split(keyword, 1)
                if len(parts) > 1:
                    topic_part = parts[1].strip()
                    # Take first few meaningful words
                    topic_words = [w for w in topic_part.split() if len(w) > 3][:3]
                    params['topic'] = ' '.join(topic_words)
                    break
        
        return params
    
    def _extract_format(self, request: str) -> str:
        """Extract document format from request."""
        request_lower = request.lower()
        if 'pdf' in request_lower:
            return 'pdf'
        elif 'docx' in request_lower or 'word' in request_lower:
            return 'docx'
        elif 'ppt' in request_lower or 'powerpoint' in request_lower:
            return 'ppt'
        return 'pdf'  # Default
    
    def _extract_email_params(self, request: str) -> Dict[str, Any]:
        """Extract email parameters from request."""
        params = {}
        request_lower = request.lower()
        
        # Try to find email addresses (simple regex-like pattern)
        import re
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', request)
        if email_match:
            params['recipient'] = email_match.group(0)
        
        # Extract format if mentioned (e.g., "send as pdf", "as ppt")
        if 'pdf' in request_lower or 'as pdf' in request_lower:
            params['format'] = 'pdf'
        elif 'docx' in request_lower or 'word' in request_lower or 'as docx' in request_lower:
            params['format'] = 'docx'
        elif 'ppt' in request_lower or 'powerpoint' in request_lower or 'as ppt' in request_lower:
            params['format'] = 'ppt'
        
        return params
    
    def execute_route(self, routing_decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the routed task.
        
        Args:
            routing_decision: Routing decision with handler and parameters
            
        Returns:
            Execution result
        """
        handler_name = routing_decision.get('handler')
        parameters = routing_decision.get('extracted_parameters', {})
        
        if handler_name not in self.handlers:
            logger.error(f"Handler not found: {handler_name}")
            return {
                "success": False,
                "error": f"Handler '{handler_name}' not registered",
                "fallback": "conversation_handler"
            }
        
        handler_function = self.handlers[handler_name]
        
        try:
            result = handler_function(parameters, routing_decision)
            return {
                "success": True,
                "handler_used": handler_name,
                "result": result
            }
        except Exception as e:
            logger.error(f"Handler execution failed: {e}")
            return {
                "success": False,
                "handler_used": handler_name,
                "error": str(e)
            }
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        total_routes = sum(self.routing_stats.values())
        return {
            "total_routes": total_routes,
            "handler_distribution": self.routing_stats,
            "registered_handlers": list(self.handlers.keys())
        }
        """Get routing statistics."""
        total_routes = sum(self.routing_stats.values())
        return {
            "total_routes": total_routes,
            "handler_distribution": self.routing_stats,
            "registered_handlers": list(self.handlers.keys())
        }

