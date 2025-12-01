"""
Enhanced Multi-Agent System for Environmental SME
Integrates prompt_library.py for all task types
Supports: QA, Quiz, Study Guide, Awareness Content, Email, Library Retrieval
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, Any, Optional, List
import json
import logging
from datetime import datetime

from parte.conversation_manager import ConversationManager
from parte.observations import ObservationsLogger
from partf.prompt_library import PromptLibrary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskDetector:
    """Detects task type from user query using LLM and pattern matching."""
    
    def __init__(self, llm_client):
        self.llm = llm_client
        self.prompt_library = PromptLibrary()
    
    def detect_task(self, user_query: str, conversation_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect task type and extract parameters.
        
        Returns:
            Dict with task_type, handler, parameters, and confidence
        """
        query_lower = user_query.lower()
        
        # Enhanced pattern matching
        task_patterns = {
            "qa": {
                "patterns": ['what is', 'what are', 'explain', 'how does', 'why', 'tell me about', 
                            'define', 'describe', 'can you explain', 'what do you know about'],
                "handler": "qa_handler",
                "confidence": "high"
            },
            "library_retrieval": {
                "patterns": ['search', 'find', 'retrieve', 'look up', 'get information', 
                            'from library', 'from documents', 'in the library'],
                "handler": "library_handler",
                "confidence": "high"
            },
            "quiz": {
                "patterns": ['quiz', 'generate quiz', 'create quiz', 'make quiz', 
                            'questions about', 'test me on'],
                "handler": "quiz_handler",
                "confidence": "high"
            },
            "study_guide": {
                "patterns": ['study guide', 'study-guide', 'studyguide', 'guide', 
                            'learning guide', 'comprehensive guide'],
                "handler": "study_guide_handler",
                "confidence": "high"
            },
            "awareness": {
                "patterns": ['article', 'awareness', 'social media', 'post', 'tweet', 
                            'infographic', 'poster', 'campaign', 'raise awareness'],
                "handler": "awareness_handler",
                "confidence": "high"
            },
            "email": {
                "patterns": ['email', 'send', 'mail', 'share via email', 'email to', 
                            'send to', '@'],
                "handler": "email_handler",
                "confidence": "high"
            },
            "document": {
                "patterns": ['export', 'download', 'save', 'pdf', 'docx', 
                            'convert to', 'generate document', 'export document', 'download document'],
                "handler": "document_handler",
                "confidence": "high"
            },
            "library_upload": {
                "patterns": ['upload', 'add to library', 'add document', 'upload document', 
                            'add to knowledge base', 'index document', 'how do i upload', 
                            'how to upload', 'upload a document', 'upload file', 'add file'],
                "handler": "library_upload_handler",
                "confidence": "high"
            },
            "awareness_email": {
                "patterns": ['awareness email', 'send awareness email', 'send awareness', 
                            'batch awareness', 'awareness campaign', 'awareness email to',
                            'send awareness to', 'awareness email about', 'send awareness email about'],
                "handler": "awareness_email_handler",
                "confidence": "high"
            },
            "combined": {
                "patterns": ['and send', 'then email', 'and email', 'generate and send',
                            'create and email', 'make and send'],
                "handler": "combined_handler",
                "confidence": "medium"
            }
        }
        
        # Check for combined tasks (e.g., "generate study guide and send as email")
        # IMPORTANT: Check specific tasks BEFORE general patterns to avoid false positives
        detected_tasks = []
        
        # First, check for specific combined tasks that should be handled as single tasks
        # CRITICAL: Check awareness_email FIRST - before any other email patterns
        # This prevents confusion with regular "send it" email tasks
        awareness_email_patterns = task_patterns["awareness_email"]["patterns"]
        # More aggressive matching for awareness emails
        has_awareness = "awareness" in query_lower
        has_email_keyword = "email" in query_lower or "@" in user_query
        has_send = "send" in query_lower
        
        # If query contains "awareness" + "email" or "send awareness", it's definitely awareness_email
        if (has_awareness and has_email_keyword) or (has_send and has_awareness):
            logger.info("Detected awareness_email task (priority check)")
            return {
                "task_type": "awareness_email",
                "handler": "awareness_email_handler",
                "confidence": "high",
                "parameters": self._extract_params(user_query, "awareness_email")
            }
        
        # Check library_upload second (before document handler to avoid false positives)
        # If query contains upload-related keywords, prioritize library_upload over document
        upload_keywords = ['upload', 'add to library', 'add document', 'how do i upload', 'how to upload']
        if any(keyword in query_lower for keyword in upload_keywords):
            # Check if it's really about uploading (not exporting)
            if 'export' not in query_lower and 'download' not in query_lower and 'save' not in query_lower:
                return {
                    "task_type": "library_upload",
                    "handler": "library_upload_handler",
                    "confidence": "high",
                    "parameters": self._extract_params(user_query, "library_upload")
                }
        
        # Otherwise, check all other patterns
        for task_name, task_info in task_patterns.items():
            # Skip combined, awareness_email, and library_upload (already checked)
            if task_name in ["combined", "awareness_email", "library_upload"]:
                continue
            if any(pattern in query_lower for pattern in task_info["patterns"]):
                detected_tasks.append({
                    "task": task_name,
                    "handler": task_info["handler"],
                    "confidence": task_info["confidence"]
                })
        
        # If multiple tasks detected, it's a combined task
        if len(detected_tasks) > 1:
            return {
                "task_type": "combined",
                "handler": "combined_handler",
                "subtasks": detected_tasks,
                "confidence": "high",
                "parameters": self._extract_combined_params(user_query, detected_tasks)
            }
        elif len(detected_tasks) == 1:
            task_info = detected_tasks[0]
            return {
                "task_type": task_info["task"],
                "handler": task_info["handler"],
                "confidence": task_info["confidence"],
                "parameters": self._extract_params(user_query, task_info["task"])
            }
        
        # Use LLM for intelligent detection
        return self._llm_detect_task(user_query, conversation_context)
    
    def _extract_params(self, query: str, task_type: str) -> Dict[str, Any]:
        """Extract parameters for specific task type."""
        params = {}
        query_lower = query.lower()
        
        if task_type == "quiz":
            # Extract topic, num_questions, difficulty
            prompt = f"""Extract quiz parameters from: "{query}"

Return JSON:
{{
  "topic": "clear topic name",
  "num_questions": number (default 5),
  "difficulty": "easy|medium|hard" (default medium)
}}"""
            try:
                result = self.llm.generate_json(prompt)
                if result:
                    params.update(result)
            except:
                pass
            
            # Fallback extraction
            if "topic" not in params:
                # Try to find topic after keywords, but stop before email/document keywords, numbers, or difficulty
                query_lower = query.lower()
                # Find position of stop keywords (email/document, numbers, difficulty, connectors)
                stop_keywords = ["email", "mail", "send", "pdf", "docx", "ppt", "to my", "to the", 
                                " and ", " then ", " with ", " at ", " difficulty", " questions", " question"]
                stop_positions = [query_lower.find(kw) for kw in stop_keywords if kw in query_lower]
                # Also check for number patterns (e.g., "2 questions", "5 questions")
                import re
                num_pattern = re.search(r'\d+\s*(?:questions?|qs?)', query_lower)
                if num_pattern:
                    stop_positions.append(num_pattern.start())
                
                stop_pos = min(stop_positions) if stop_positions else len(query)
                
                # Extract topic from the beginning up to stop position
                topic_query = query[:stop_pos].lower()
                
                # Try to find topic after keywords
                for keyword in ["about", "on", "regarding", "for"]:
                    if keyword in topic_query:
                        parts = topic_query.split(keyword, 1)
                        if len(parts) > 1:
                            topic = parts[1].strip()
                            # Remove quiz-related words
                            for word in ["quiz", "questions", "generate", "create", "make", "a", "an", "the"]:
                                topic = topic.replace(word, " ").strip()
                            # Clean up extra spaces
                            topic = " ".join(topic.split())
                            if topic and len(topic) > 3:
                                params["topic"] = topic
                                break
                
                # If still no topic, try extracting from the beginning (before "and", "then", "with", etc.)
                if "topic" not in params:
                    # Split on common connectors and parameters
                    for connector in [" and ", " then ", " and send", " and email", " with ", " at "]:
                        if connector in topic_query:
                            topic_query = topic_query.split(connector)[0]
                            break
                    
                    # Remove generation keywords
                    for word in ["generate", "create", "make", "quiz", "a", "an", "the", "on", "about"]:
                        topic_query = topic_query.replace(word, " ").strip()
                    
                    topic_query = " ".join(topic_query.split())  # Clean up spaces
                    if topic_query and len(topic_query) > 3:
                        params["topic"] = topic_query
                    else:
                        params["topic"] = "sustainability"  # Default
            
            # Extract num_questions
            import re
            num_match = re.search(r'(\d+)\s*(?:questions?|qs?)', query_lower)
            if num_match:
                params["num_questions"] = int(num_match.group(1))
            else:
                params["num_questions"] = 5
            
            # Extract difficulty - check for explicit mentions first
            if "medium" in query_lower or "moderate" in query_lower:
                params["difficulty"] = "medium"
            elif "easy" in query_lower:
                params["difficulty"] = "easy"
            elif "hard" in query_lower or "difficult" in query_lower:
                params["difficulty"] = "hard"
            else:
                # Default to medium if not specified
                params["difficulty"] = "medium"
            
            # Override with LLM result if it exists and is valid
            if "difficulty" in params and params["difficulty"] not in ["easy", "medium", "hard"]:
                # If LLM returned invalid difficulty, use fallback
                if "medium" in query_lower:
                    params["difficulty"] = "medium"
                elif "easy" in query_lower:
                    params["difficulty"] = "easy"
                elif "hard" in query_lower or "difficult" in query_lower:
                    params["difficulty"] = "hard"
                else:
                    params["difficulty"] = "medium"
        
        elif task_type == "study_guide":
            # Extract topic
            for keyword in ["about", "on", "regarding", "for"]:
                if keyword in query_lower:
                    parts = query_lower.split(keyword, 1)
                    if len(parts) > 1:
                        topic = parts[1].strip()
                        for word in ["study guide", "guide", "generate", "create"]:
                            topic = topic.replace(word, "").strip()
                        if topic:
                            params["topic"] = topic
                            break
            if "topic" not in params:
                params["topic"] = "sustainability"
        
        elif task_type == "awareness":
            # Extract format and topic
            if "article" in query_lower:
                params["format"] = "article"
            elif "social media" in query_lower or "post" in query_lower:
                params["format"] = "social_media"
            elif "infographic" in query_lower:
                params["format"] = "infographic_text"
            elif "poster" in query_lower:
                params["format"] = "poster_text"
            else:
                params["format"] = "article"
            
            # Extract topic
            for keyword in ["about", "on", "regarding"]:
                if keyword in query_lower:
                    parts = query_lower.split(keyword, 1)
                    if len(parts) > 1:
                        params["topic"] = parts[1].strip()
                        break
            if "topic" not in params:
                params["topic"] = "environmental sustainability"
        
        elif task_type == "email":
            # Extract recipient
            import re
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', query)
            if email_match:
                params["recipient"] = email_match.group(0)
            # Also check for format if mentioned (e.g., "send as pdf")
            query_lower = query.lower()
            if "pdf" in query_lower:
                params["format"] = "pdf"
            elif "docx" in query_lower or "word" in query_lower:
                params["format"] = "docx"
        
        elif task_type == "document":
            # Extract format
            query_lower = query.lower()
            if "pdf" in query_lower:
                params["format"] = "pdf"
            elif "docx" in query_lower or "word" in query_lower:
                params["format"] = "docx"
            else:
                params["format"] = "pdf"  # Default
        
        elif task_type == "qa" or task_type == "library_retrieval":
            params["question"] = query
        
        elif task_type == "library_upload":
            # Extract file name or document reference if mentioned
            query_lower = query.lower()
            # Try to find document name
            for keyword in ["document", "file", "pdf", "docx"]:
                if keyword in query_lower:
                    # Extract text after keyword
                    parts = query_lower.split(keyword, 1)
                    if len(parts) > 1:
                        doc_name = parts[1].strip()
                        # Remove common words
                        for word in ["named", "called", "titled", "about", "on"]:
                            doc_name = doc_name.replace(word, "").strip()
                        if doc_name:
                            params["document_name"] = doc_name
                            break
        
        elif task_type == "awareness_email":
            # Extract topic - find text between "about/on" and "to" (before email addresses)
            query_lower = query.lower()
            import re
            
            # Find position of "to" before email addresses
            to_pos = query_lower.find(" to ")
            if to_pos == -1:
                to_pos = len(query)
            
            # Extract topic part (before "to")
            topic_query = query[:to_pos].lower()
            
            # Try to find topic after keywords
            for keyword in ["about", "on", "regarding"]:
                if keyword in topic_query:
                    parts = topic_query.split(keyword, 1)
                    if len(parts) > 1:
                        topic = parts[1].strip()
                        # Remove email-related words
                        for word in ["email", "send", "awareness", "campaign", "the"]:
                            topic = topic.replace(word, "").strip()
                        # Clean up
                        topic = " ".join(topic.split())
                        if topic and len(topic) > 3:
                            params["topic"] = topic
                            break
            
            if "topic" not in params:
                # Try extracting from beginning if no "about/on" found
                # Remove "send awareness email" and extract what's left before "to"
                topic_query = topic_query.replace("send awareness email", "").replace("awareness email", "").strip()
                if topic_query and len(topic_query) > 3:
                    params["topic"] = topic_query
                else:
                    params["topic"] = "environmental sustainability"
            
            # Extract audience if mentioned - check before email addresses
            # Look for audience patterns like "to [audience] at" or "to [audience]" before email
            audience_keywords = {
                "campus college students": ["campus college students", "college students", "campus students"],
                "college students": ["college students", "university students"],
                "students": ["students", "student", "learners"],
                "teachers": ["teachers", "educators", "instructors"],
                "general public": ["general public", "public", "everyone"],
                "professionals": ["professionals", "experts", "practitioners"],
                "community": ["community", "communities"],
                "researchers": ["researchers", "scientists", "academics"]
            }
            
            # Find position of "to" before email addresses
            to_pos = query_lower.find(" to ")
            if to_pos == -1:
                to_pos = len(query)
            
            # Check audience in the part before email addresses
            audience_query = query_lower[:to_pos]
            
            # Check for specific audience types (longer phrases first)
            for audience_type, keywords in audience_keywords.items():
                if any(kw in audience_query for kw in keywords):
                    params["audience"] = audience_type
                    logger.info(f"Extracted audience: {audience_type}")
                    break
            
            if "audience" not in params:
                params["audience"] = "general public"
        
        return params
    
    def _extract_combined_params(self, query: str, detected_tasks: List[Dict]) -> Dict[str, Any]:
        """Extract parameters for combined tasks."""
        params = {}
        
        # Extract parameters for each subtask
        for task_info in detected_tasks:
            task_type = task_info["task"]
            task_params = self._extract_params(query, task_type)
            
            # For quiz/study_guide, if topic is missing, try to extract from the full query
            if task_type in ["quiz", "study_guide"] and "topic" not in task_params:
                # Try to find topic before email/document keywords, numbers, or difficulty
                query_lower = query.lower()
                # Find position of stop keywords
                stop_keywords = ["email", "mail", "send", "pdf", "docx", "ppt", " and ", " then ", 
                                " with ", " at ", " difficulty", " questions", " question"]
                stop_positions = [query_lower.find(kw) for kw in stop_keywords if kw in query_lower]
                # Also check for number patterns
                import re
                num_pattern = re.search(r'\d+\s*(?:questions?|qs?)', query_lower)
                if num_pattern:
                    stop_positions.append(num_pattern.start())
                
                stop_pos = min(stop_positions) if stop_positions else len(query)
                
                # Extract topic from the beginning up to stop position
                topic_part = query[:stop_pos].lower()
                
                # Try to find topic after "on" or "about"
                for keyword in ["on", "about"]:
                    if keyword in topic_part:
                        parts = topic_part.split(keyword, 1)
                        if len(parts) > 1:
                            topic = parts[1].strip()
                            # Remove generation keywords
                            for word in ["generate", "create", "make", "quiz", "study guide", "a", "an", "the"]:
                                topic = topic.replace(word, " ").strip()
                            topic = " ".join(topic.split())  # Clean up spaces
                            if topic and len(topic) > 3:
                                task_params["topic"] = topic
                                break
                
                # If still no topic, remove generation keywords from beginning
                if "topic" not in task_params:
                    for keyword in ["generate", "create", "make", "quiz", "study guide", "on", "about", "a", "an", "the"]:
                        topic_part = topic_part.replace(keyword, " ").strip()
                    topic_part = " ".join(topic_part.split())  # Clean up spaces
                    if topic_part and len(topic_part) > 3:
                        task_params["topic"] = topic_part
                    else:
                        task_params["topic"] = "sustainability"  # Default
            
            params[task_type] = task_params
        
        return params
    
    def _llm_detect_task(self, query: str, context: Optional[str]) -> Dict[str, Any]:
        """Use LLM for intelligent task detection."""
        prompt = f"""Analyze this user query and determine the task type:

Query: "{query}"
Context: {context or "None"}

Task Types:
1. qa - Question answering (what, how, why, explain, tell me)
2. library_retrieval - Search/retrieve from library/documents
3. quiz - Generate quiz questions
4. study_guide - Generate study guide
5. awareness - Generate awareness content (article, social media, etc.)
6. email - Send email
7. document - Export/download document
8. combined - Multiple tasks (e.g., generate and send)

Return JSON:
{{
  "task_type": "task_type_name",
  "handler": "handler_name",
  "confidence": "high|medium|low",
  "parameters": {{"key": "value"}},
  "reasoning": "why this task type"
}}"""
        
        try:
            result = self.llm.generate_json(prompt)
            if result and "task_type" in result:
                return result
        except Exception as e:
            logger.warning(f"LLM task detection failed: {e}")
        
        # Default to QA
        return {
            "task_type": "qa",
            "handler": "qa_handler",
            "confidence": "low",
            "parameters": {"question": query}
        }


class EnhancedMultiAgent:
    """
    Enhanced Multi-Agent System that uses prompt_library.py
    Handles all task types: QA, Quiz, Study Guide, Awareness, Email, Library Retrieval
    """
    
    def __init__(self, llm_client, rag_system, content_generator, knowledge_tools=None):
        """
        Initialize enhanced multi-agent system.
        
        Args:
            llm_client: LLM client
            rag_system: RAG system for QA and retrieval
            content_generator: Content generator for quizzes, guides, awareness
            knowledge_tools: Knowledge tools for library retrieval
        """
        self.llm = llm_client
        self.rag = rag_system
        self.content_gen = content_generator
        self.knowledge_tools = knowledge_tools
        
        # Initialize components
        self.prompt_library = PromptLibrary()
        self.task_detector = TaskDetector(llm_client)
        self.conversation = ConversationManager()
        self.observations = ObservationsLogger()
        
        logger.info("✓ Enhanced Multi-Agent System initialized")
    
    def _handle_context_aware_request(self, user_request: str) -> Optional[Dict[str, Any]]:
        """
        Handle context-aware requests that reference previously generated content.
        
        Examples:
        - "make it into a document"
        - "send it to my email"
        - "convert it to pdf"
        - "download it as docx"
        """
        query_lower = user_request.lower()
        
        # DON'T trigger if the query contains generation keywords (let combined handler deal with it)
        generation_keywords = ["generate", "create", "make a", "make an", "create a", "create an"]
        if any(keyword in query_lower for keyword in generation_keywords):
            return None
        
        # CRITICAL: Don't trigger context-aware handler for awareness email requests
        # Awareness emails are their own task type, not follow-ups to generated content
        if "awareness" in query_lower and ("email" in query_lower or "@" in user_request):
            logger.info("Skipping context-aware handler for awareness email request")
            return None
        
        # Check if user is referencing previous content
        # Prioritize "send it" patterns - these are very common after quiz/study guide generation
        reference_patterns = [
            "send it", "send it as", "email it", "mail it",  # Most common - check first
            "make it", "convert it", "turn it", "change it",
            "send the", "send to", "send as",
            "download it", "save it", "export it",
            "make this", "convert this", "send this", "email this",
            "can you send", "please send",
            "send the mail", "send via email", "email to",
            "email as", "email it as",
            "to my email", "to email", "to the email"
        ]
        
        # Also check for email address pattern (indicates user wants to send something)
        import re
        has_email = bool(re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_request))
        has_send_keywords = any(word in query_lower for word in ["send", "email", "mail"])
        has_format_keywords = any(word in query_lower for word in ["pdf", "docx", "ppt", "powerpoint", "document"])
        
        # CRITICAL: If user says "send it" or "send it as" with email, this is definitely context-aware
        if ("send it" in query_lower or "send it as" in query_lower) and has_email:
            logger.info("Detected 'send it' pattern with email - treating as context-aware request")
        # If user mentions email address and send/format keywords, treat as context-aware
        elif has_email and (has_send_keywords or has_format_keywords):
            # This is likely a follow-up request - continue to check cached content
            logger.info("Detected email address with send/format keywords - treating as context-aware request")
        elif not any(pattern in query_lower for pattern in reference_patterns):
            return None
        
        # Check for cached content
        last_content = self.conversation.get_cached_context("last_generated_content")
        if not last_content:
            return {
                "success": False,
                "type": "error",
                "message": "I don't see any previously generated content. Please generate a quiz or study guide first, then I can help you export or send it."
            }
        
        # Determine what the user wants to do
        wants_document = any(word in query_lower for word in ["document", "pdf", "docx", "ppt", "download", "save", "export", "convert"])
        wants_email = any(word in query_lower for word in ["email", "mail", "send"])
        
        # Extract format - check for "as [format]" or "[format]" patterns
        format_type = "pdf"  # Default
        query_lower = user_request.lower()
        
        # Check for "as pdf/docx/ppt" pattern first (more specific)
        if " as pdf" in query_lower or "as a pdf" in query_lower:
            format_type = "pdf"
        elif " as docx" in query_lower or "as a docx" in query_lower or " as word" in query_lower:
            format_type = "docx"
        elif " as ppt" in query_lower or "as a ppt" in query_lower or " as powerpoint" in query_lower:
            format_type = "ppt"
        # Fallback to checking for format keywords anywhere
        elif "pdf" in query_lower:
            format_type = "pdf"
        elif "docx" in query_lower or "word" in query_lower:
            format_type = "docx"
        elif "ppt" in query_lower or "powerpoint" in query_lower:
            format_type = "ppt"
        
        logger.info(f"Extracted format: {format_type}")
        
        # Extract email addresses if mentioned (can be multiple, but for context-aware we use first)
        email_addresses = []
        email_address = None
        if wants_email:
            import re
            email_matches = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', user_request)
            email_addresses = email_matches
            email_address = email_addresses[0] if email_addresses else None
        
        results = []
        
        # Handle document export if requested
        if wants_document:
            logger.info("📄 Handling context-aware document export...")
            try:
                from partd.export_tools import DocumentExporter
                exporter = DocumentExporter()
                
                content_type = last_content.get("content_type", "document")
                quiz_data = last_content.get("quiz")
                # Get study guide data - it might be nested or at root level
                study_guide_data = last_content.get("study_guide")
                if not study_guide_data and content_type == "study_guide":
                    # If study_guide key doesn't exist, try to get content directly
                    study_guide_data = last_content.get("content")
                    if study_guide_data:
                        # Wrap it in a dict if it's just a string
                        if isinstance(study_guide_data, str):
                            study_guide_data = {
                                "content": study_guide_data,
                                "topic": last_content.get("topic", "study guide")
                            }
                
                if content_type == "quiz" and quiz_data:
                    if format_type == "pdf":
                        filepath = exporter.export_quiz_to_pdf(quiz_data)
                    elif format_type == "docx":
                        filepath = exporter.export_quiz_to_docx(quiz_data)
                    else:
                        # PPT not supported for quiz, default to PDF
                        filepath = exporter.export_quiz_to_pdf(quiz_data)
                        format_type = "pdf"
                elif content_type == "study_guide" and study_guide_data:
                    # Check if study_guide_data is a dict with content or just a string
                    if isinstance(study_guide_data, dict):
                        content_text = study_guide_data.get("content", "")
                        topic = study_guide_data.get("topic", last_content.get("topic", "study guide"))
                    else:
                        content_text = study_guide_data
                        topic = last_content.get("topic", "study guide")
                    
                    if format_type == "ppt" or format_type == "pptx":
                        # Export to PPT
                        try:
                            filepath = exporter.export_content_to_ppt(
                                {"content": content_text, "topic": topic},
                                "study_guide"
                            )
                        except Exception as e:
                            logger.warning(f"PPT export failed: {e}, falling back to DOCX")
                            filepath = exporter.export_content_to_docx(
                                {"content": content_text, "topic": topic},
                                "study_guide"
                            )
                            format_type = "docx"
                    elif format_type == "pdf":
                        # For PDF, export as DOCX first (can be converted later)
                        filepath = exporter.export_content_to_docx(
                            {"content": content_text, "topic": topic},
                            "study_guide"
                        )
                        format_type = "docx"  # For now, use docx as PDF conversion requires additional libraries
                    else:
                        filepath = exporter.export_content_to_docx(
                            {"content": content_text, "topic": topic},
                            "study_guide"
                        )
                else:
                    return {
                        "success": False,
                        "type": "error",
                        "message": "Could not export content. The cached content format is not supported."
                    }
                
                # Cache filepath
                self.conversation.cache_context("last_exported_file", filepath)
                last_content["filepath"] = filepath
                self.conversation.cache_context("last_generated_content", last_content)
                
                results.append({
                    "success": True,
                    "type": "document",
                    "message": f"Document exported to {filepath}",
                    "filepath": filepath,
                    "format": format_type
                })
                
            except Exception as e:
                logger.error(f"Context-aware document export failed: {e}")
                results.append({
                    "success": False,
                    "type": "error",
                    "message": f"Failed to export document: {str(e)}"
                })
        
        # Handle email if requested
        if wants_email:
            logger.info("📧 Handling context-aware email...")
            if not email_address:
                return {
                    "success": False,
                    "type": "error",
                    "message": "Please provide an email address. Example: 'send it to user@example.com'"
                }
            
            filepath = self.conversation.get_cached_context("last_exported_file")
            if not filepath:
                filepath = last_content.get("filepath", "")
            
            # If no filepath exists, automatically export the document first
            if not filepath:
                logger.info("📄 No exported file found, exporting document automatically...")
                try:
                    from partd.export_tools import DocumentExporter
                    exporter = DocumentExporter()
                    
                    content_type = last_content.get("content_type", "document")
                    quiz_data = last_content.get("quiz")
                    study_guide_data = last_content.get("study_guide")
                    if not study_guide_data and content_type == "study_guide":
                        study_guide_data = last_content.get("content")
                    
                    if content_type == "quiz" and quiz_data:
                        # Default to PDF for email
                        filepath = exporter.export_quiz_to_pdf(quiz_data)
                    elif content_type == "study_guide" and study_guide_data:
                        if isinstance(study_guide_data, dict):
                            content_text = study_guide_data.get("content", "")
                            topic = study_guide_data.get("topic", last_content.get("topic", "study guide"))
                        else:
                            content_text = study_guide_data
                            topic = last_content.get("topic", "study guide")
                        filepath = exporter.export_content_to_docx(
                            {"content": content_text, "topic": topic},
                            "study_guide"
                        )
                    else:
                        return {
                            "success": False,
                            "type": "error",
                            "message": "Could not export content. The cached content format is not supported."
                        }
                    
                    # Cache the filepath
                    self.conversation.cache_context("last_exported_file", filepath)
                    last_content["filepath"] = filepath
                    self.conversation.cache_context("last_generated_content", last_content)
                    
                    results.append({
                        "success": True,
                        "type": "document",
                        "message": f"Document exported to {filepath}",
                        "filepath": filepath,
                        "format": format_type  # Use the requested format
                    })
                    
                except Exception as e:
                    logger.error(f"Auto-export failed: {e}")
                    return {
                        "success": False,
                        "type": "error",
                        "message": f"Failed to export document: {str(e)}"
                    }
            
            try:
                result = self.content_gen.send_content_via_email(
                    recipient=email_address,
                    content_filepath=filepath,
                    content_type=last_content.get("content_type", "document"),
                    topic=last_content.get("topic", "content")
                )
                
                results.append({
                    "success": result.get("sent", False),
                    "type": "email",
                    "message": f"Mail sent to {email_address}" if result.get('sent') else f"Failed to send email to {email_address}",
                    "response": f"Mail sent to {email_address}" if result.get('sent') else f"Failed to send email to {email_address}",
                    "metadata": result
                })
            except Exception as e:
                logger.error(f"Context-aware email failed: {e}")
                results.append({
                    "success": False,
                    "type": "error",
                    "message": f"Failed to send email: {str(e)}"
                })
        
        # Combine results
        if not results:
            return None
        
        success_count = sum(1 for r in results if r.get("success"))
        
        # Create user-friendly message
        email_result = next((r for r in results if r.get("type") == "email"), None)
        if email_result and email_result.get("success"):
            # If email was sent, use a friendly message
            recipient_email = email_result.get("metadata", {}).get("recipient") or email_address or "your email"
            message = f"Mail sent to {recipient_email}"
        else:
            message = f"Completed {success_count}/{len(results)} task(s)"
        
        # Return appropriate response
        if len(results) == 1:
            return results[0]
        else:
            return {
                "success": success_count > 0,
                "type": "combined",
                "message": message,
                "response": message,  # Also set response for UI
                "results": results,
                "metadata": {"num_subtasks": len(results), "completed": success_count}
            }
    
    def process_request(self, user_request: str) -> Dict[str, Any]:
        """
        Process user request with multi-agent system.
        
        Args:
            user_request: User's request
            
        Returns:
            Agent response
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing request: {user_request}")
        logger.info(f"{'='*70}")
        
        # Add to conversation
        self.conversation.add_message("user", user_request)
        
        # CRITICAL: Check for context-aware requests FIRST (references to previously generated content)
        # This handles "send it" patterns that come after quiz/study guide generation
        context_aware_result = self._handle_context_aware_request(user_request)
        if context_aware_result is not None:
            logger.info("Handled as context-aware request")
            return context_aware_result
        
        # Detect task
        context = self.conversation.get_formatted_history(3)
        task_info = self.task_detector.detect_task(user_request, context)
        
        logger.info(f"Detected task: {task_info['task_type']} (handler: {task_info['handler']})")
        
        # Route to appropriate handler
        handler = task_info["handler"]
        parameters = task_info.get("parameters", {})
        
        try:
            if handler == "qa_handler":
                result = self._handle_qa(user_request, parameters)
            elif handler == "library_handler":
                result = self._handle_library_retrieval(user_request, parameters)
            elif handler == "quiz_handler":
                result = self._handle_quiz(user_request, parameters)
            elif handler == "study_guide_handler":
                result = self._handle_study_guide(user_request, parameters)
            elif handler == "awareness_handler":
                result = self._handle_awareness(user_request, parameters)
            elif handler == "email_handler":
                result = self._handle_email(user_request, parameters)
            elif handler == "document_handler":
                result = self._handle_document(user_request, parameters)
            elif handler == "library_upload_handler":
                result = self._handle_library_upload(user_request, parameters)
            elif handler == "awareness_email_handler":
                result = self._handle_awareness_email(user_request, parameters)
            elif handler == "combined_handler":
                result = self._handle_combined(user_request, parameters, task_info)
            else:
                # Default to QA
                result = self._handle_qa(user_request, {"question": user_request})
            
            # Cache content for follow-up requests
            if result.get("success"):
                if "quiz" in result:
                    self.conversation.cache_context("last_generated_content", {
                        "content_type": "quiz",
                        "topic": result["quiz"].get("topic", ""),
                        "quiz": result["quiz"]
                    })
                elif "study_guide" in result:
                    self.conversation.cache_context("last_generated_content", {
                        "content_type": "study_guide",
                        "topic": result["study_guide"].get("topic", ""),
                        "study_guide": result["study_guide"],
                        "content": result["study_guide"].get("content", "")
                    })
                elif "awareness_content" in result:
                    self.conversation.cache_context("last_generated_content", {
                        "content_type": "awareness",
                        "topic": result["awareness_content"].get("topic", ""),
                        "awareness_content": result["awareness_content"]
                    })
                elif result.get("type") == "awareness_email":
                    # Cache awareness email metadata
                    self.conversation.cache_context("last_awareness_email", {
                        "topic": result.get("metadata", {}).get("topic", ""),
                        "recipients": result.get("metadata", {}).get("total_recipients", 0),
                        "successful": result.get("metadata", {}).get("successful", 0)
                    })
            
            # Add to conversation
            response_message = result.get("message", result.get("response", ""))
            self.conversation.add_message("agent", response_message)
            
            # Ensure all fields are included in the response
            final_result = {
                **result,
                "task_type": task_info["task_type"],
                "handler": handler,
                "metadata": {
                    **result.get("metadata", {}),
                    "detection_confidence": task_info.get("confidence", "medium")
                }
            }
            
            # Ensure success is True if not explicitly False
            if "success" not in final_result or final_result.get("success") is None:
                final_result["success"] = True
            
            # Ensure response/message fields are set
            if not final_result.get("response") and final_result.get("message"):
                final_result["response"] = final_result["message"]
            if not final_result.get("message") and final_result.get("response"):
                final_result["message"] = final_result["response"]
            
            return final_result
        
        except Exception as e:
            logger.error(f"Handler execution failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "type": "error",
                "message": f"I encountered an error: {str(e)}. Please try rephrasing your request.",
                "error": str(e)
            }
    
    def _handle_qa(self, query: str, params: Dict) -> Dict[str, Any]:
        """Handle Q&A using RAG system - retrieved chunks are passed to LLM."""
        logger.info("📚 Handling Q&A...")
        
        question = params.get("question", query)
        
        # Use RAG system which:
        # 1. Retrieves top-k chunks from library/web
        # 2. Builds context from retrieved chunks
        # 3. Passes context + question to LLM with proper prompt structure
        # 4. Generates answer using retrieved information
        # The RAG system's answer_question() method already implements proper RAG:
        # - Retrieves chunks
        # - Builds context string from chunks
        # - Passes context + question to LLM with instructions to use retrieved info
        # - Generates answer based on retrieved chunks
        rag_result = self.rag.answer_question(question, top_k=5)
        
        # The RAG system already generates the answer using retrieved chunks
        answer = rag_result.get("answer", "")
        sources = rag_result.get("sources", {})
        
        # Ensure sources are properly formatted
        # RAG returns: {"local": [...], "web": [...]}
        # Ensure local and web are lists
        if not isinstance(sources, dict):
            sources = {}
        
        local_sources = sources.get("local", [])
        web_sources = sources.get("web", [])
        
        # Ensure sources have the right format for UI
        formatted_sources = {
            "local": local_sources if isinstance(local_sources, list) else [],
            "web": web_sources if isinstance(web_sources, list) else []
        }
        
        return {
            "success": True,
            "type": "answer",
            "message": answer,
            "response": answer,
            "sources": formatted_sources,
            "metadata": {
                "used_rag": True,
                "retrieval_confidence": rag_result.get("metadata", {}).get("retrieval_confidence", 0),
                "used_web_search": rag_result.get("metadata", {}).get("used_web_search", False),
                "num_local_sources": len(formatted_sources["local"]),
                "num_web_sources": len(formatted_sources["web"])
            }
        }
    
    def _handle_library_retrieval(self, query: str, params: Dict) -> Dict[str, Any]:
        """Handle library retrieval using knowledge tools."""
        logger.info("📖 Handling library retrieval...")
        
        if not self.knowledge_tools:
            # Fallback to RAG
            return self._handle_qa(query, {"question": query})
        
        # Use knowledge tools for retrieval
        result = self.knowledge_tools.retrieve_knowledge(
            query=query,
            method="hybrid",
            top_k=5
        )
        
        if result.get("success"):
            # Format results
            results_text = "Library Retrieval Results:\n\n"
            for i, item in enumerate(result.get("results", [])[:5], 1):
                results_text += f"{i}. {item.get('content', '')[:200]}...\n"
                results_text += f"   Source: {item.get('source', 'Unknown')}\n"
                results_text += f"   Score: {item.get('score', 0):.3f}\n\n"
            
            return {
                "success": True,
                "type": "library_retrieval",
                "message": results_text,
                "response": results_text,
                "results": result.get("results", []),
                "metadata": {
                    "num_results": len(result.get("results", [])),
                    "method": result.get("method", "hybrid")
                }
            }
        else:
            return {
                "success": False,
                "type": "error",
                "message": "Could not retrieve information from library. Please try a different query.",
                "error": result.get("error", "Unknown error")
            }
    
    def _handle_quiz(self, query: str, params: Dict) -> Dict[str, Any]:
        """Handle quiz generation using prompt library."""
        logger.info("📝 Handling quiz generation...")
        
        topic = params.get("topic", "sustainability")
        num_questions = params.get("num_questions", 5)
        difficulty = params.get("difficulty", "medium")
        
        # Normalize difficulty
        difficulty_lower = str(difficulty).lower()
        if "medium" in difficulty_lower or "moderate" in difficulty_lower:
            difficulty = "medium"
        elif "easy" in difficulty_lower:
            difficulty = "easy"
        elif "hard" in difficulty_lower or "difficult" in difficulty_lower:
            difficulty = "hard"
        else:
            difficulty = "medium"  # Default to medium
        
        # Get context from RAG
        info_query = f"Provide comprehensive information about {topic} for quiz generation"
        rag_result = self.rag.answer_question(info_query)
        context = rag_result.get("answer", "")[:1500]
        
        # Use prompt library
        prompt = self.prompt_library.get_prompt(
            "quiz_generation",
            "difficulty_aware",
            num_questions=num_questions,
            topic=topic,
            difficulty=difficulty
        )
        
        # Add context
        full_prompt = f"{prompt}\n\nContext: {context}\n\nGenerate:"
        
        # Generate quiz
        quiz_json = self.llm.generate_json(full_prompt)
        
        # Parse and normalize quiz
        if isinstance(quiz_json, list):
            questions = quiz_json
        elif isinstance(quiz_json, dict) and "questions" in quiz_json:
            questions = quiz_json["questions"]
        else:
            # Fallback to content generator
            quiz_result = self.content_gen.generate_quiz(topic, num_questions, difficulty)
            return {
                "success": True,
                "type": "quiz",
                "message": f"Generated quiz on {topic}",
                "quiz": quiz_result,
                "metadata": {"used_prompt_library": False, "used_fallback": True}
            }
        
        # Normalize questions
        normalized_questions = []
        for q in questions[:num_questions]:
            if isinstance(q.get("options"), dict):
                options = q["options"]
            else:
                options = {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}
            
            normalized_questions.append({
                "question": q.get("question", ""),
                "options": options,
                "correct_answer": q.get("correct_answer", "A"),
                "explanation": q.get("explanation", ""),
                "difficulty": difficulty
            })
        
        quiz_result = {
            "topic": topic,
            "num_questions": len(normalized_questions),
            "difficulty": difficulty,
            "questions": normalized_questions
        }
        
        # Cache the quiz for follow-up requests
        self.conversation.cache_context("last_generated_content", {
            "content_type": "quiz",
            "topic": topic,
            "quiz": quiz_result
        })
        
        # Auto-export quiz to PDF for easy access (user can request other formats later)
        try:
            from partd.export_tools import DocumentExporter
            exporter = DocumentExporter()
            filepath = exporter.export_quiz_to_pdf(quiz_result)
            # Cache the exported file path
            self.conversation.cache_context("last_exported_file", filepath)
            # Update cached content with filepath
            cached_content = self.conversation.get_cached_context("last_generated_content")
            if cached_content:
                cached_content["filepath"] = filepath
                self.conversation.cache_context("last_generated_content", cached_content)
            logger.info(f"Auto-exported quiz to: {filepath}")
        except Exception as e:
            logger.warning(f"Auto-export failed (user can export manually): {e}")
        
        return {
            "success": True,
            "type": "quiz",
            "message": f"Generated {len(normalized_questions)}-question quiz on {topic}",
            "response": f"Generated quiz on {topic} with {len(normalized_questions)} questions at {difficulty} difficulty",
            "quiz": quiz_result,
            "metadata": {
                "used_prompt_library": True,
                "prompt_variant": "difficulty_aware"
            }
        }
    
    def _handle_study_guide(self, query: str, params: Dict) -> Dict[str, Any]:
        """Handle study guide generation using prompt library."""
        logger.info("📚 Handling study guide generation...")
        
        topic = params.get("topic", "sustainability")
        
        # Get context from RAG
        info_query = f"Provide comprehensive educational content about {topic}"
        rag_result = self.rag.answer_question(info_query)
        context = rag_result.get("answer", "")[:2000]
        
        # Use prompt library
        prompt = self.prompt_library.get_prompt(
            "study_guide",
            "comprehensive",
            topic=topic
        )
        
        # Add context
        full_prompt = f"{prompt}\n\nContext: {context}\n\nStudy Guide:"
        
        # Generate study guide
        guide_content = self.llm.generate(full_prompt, max_tokens=3000, temperature=0.7)
        
        study_guide = {
            "topic": topic,
            "content": guide_content
        }
        
        # Cache the study guide for follow-up requests
        self.conversation.cache_context("last_generated_content", {
            "content_type": "study_guide",
            "topic": topic,
            "study_guide": study_guide,
            "content": guide_content
        })
        
        # Auto-export study guide to PDF for easy access
        try:
            from partd.export_tools import DocumentExporter
            exporter = DocumentExporter()
            filepath = exporter.export_study_guide_to_pdf(study_guide)
            # Cache the exported file path
            self.conversation.cache_context("last_exported_file", filepath)
            # Update cached content with filepath
            cached_content = self.conversation.get_cached_context("last_generated_content")
            if cached_content:
                cached_content["filepath"] = filepath
                self.conversation.cache_context("last_generated_content", cached_content)
            logger.info(f"Auto-exported study guide to: {filepath}")
        except Exception as e:
            logger.warning(f"Auto-export failed (user can export manually): {e}")
        
        return {
            "success": True,
            "type": "study_guide",
            "message": f"Generated study guide on {topic}",
            "response": f"Generated study guide on {topic}",
            "study_guide": study_guide,
            "metadata": {
                "used_prompt_library": True,
                "prompt_variant": "comprehensive"
            }
        }
    
    def _handle_awareness(self, query: str, params: Dict) -> Dict[str, Any]:
        """Handle awareness content generation using prompt library."""
        logger.info("📢 Handling awareness content generation...")
        
        topic = params.get("topic", "environmental sustainability")
        format_type = params.get("format", "article")
        audience = params.get("audience", "general public")
        
        # Get context from RAG
        info_query = f"Provide engaging information about {topic} for {audience} awareness"
        rag_result = self.rag.answer_question(info_query)
        context = rag_result.get("answer", "")[:1500]
        
        # Map format to prompt library task
        if format_type == "article":
            prompt = self.prompt_library.get_prompt(
                "awareness_content",
                "article",
                word_count=500,
                topic=topic,
                audience=audience,
                tone="engaging and informative"
            )
        elif format_type == "social_media":
            prompt = self.prompt_library.get_prompt(
                "awareness_content",
                "social_media",
                num_posts=5,
                topic=topic
            )
        else:
            # Use article as default
            prompt = self.prompt_library.get_prompt(
                "awareness_content",
                "article",
                word_count=500,
                topic=topic,
                audience=audience,
                tone="engaging and informative"
            )
        
        # Add context
        full_prompt = f"{prompt}\n\nContext: {context}\n\nContent:"
        
        # Generate content
        content = self.llm.generate(full_prompt, max_tokens=2000, temperature=0.8)
        
        awareness_content = {
            "topic": topic,
            "format": format_type,
            "audience": audience,
            "content": content
        }
        
        return {
            "success": True,
            "type": "awareness",
            "message": f"Generated {format_type} on {topic}",
            "awareness_content": awareness_content,
            "metadata": {
                "used_prompt_library": True,
                "format": format_type
            }
        }
    
    def _handle_email(self, query: str, params: Dict) -> Dict[str, Any]:
        """Handle email sending."""
        logger.info("📧 Handling email...")
        
        recipient = params.get("recipient")
        if not recipient:
            return {
                "success": False,
                "type": "error",
                "message": "Please provide an email address. Example: 'Send this to user@example.com'"
            }
        
        # Get last exported file first (from document export)
        filepath = self.conversation.get_cached_context("last_exported_file")
        
        # If no exported file, get last generated content
        if not filepath:
            last_content = self.conversation.get_cached_context("last_generated_content")
            if last_content:
                filepath = last_content.get("filepath", "")
                content_type = last_content.get("content_type", "document")
                topic = last_content.get("topic", "sustainability")
            else:
                return {
                    "success": False,
                    "type": "error",
                    "message": "No content available to send. Please generate and export content first (quiz, study guide, etc.)."
                }
        else:
            # Get content type and topic from cached content
            last_content = self.conversation.get_cached_context("last_generated_content")
            if last_content:
                content_type = last_content.get("content_type", "document")
                topic = last_content.get("topic", "sustainability")
            else:
                content_type = "document"
                topic = "sustainability"
        
        if not filepath:
            return {
                "success": False,
                "type": "error",
                "message": "No document file available to send. Please export the content first."
            }
        
        # Use content generator's email functionality
        try:
            result = self.content_gen.send_content_via_email(
                recipient=recipient,
                content_filepath=filepath,
                content_type=content_type,
                topic=topic
            )
            
            return {
                "success": result.get("sent", False),
                "type": "email",
                "message": f"Mail sent to {recipient}" if result.get('sent') else f"Failed to send email to {recipient}",
                "response": f"Mail sent to {recipient}" if result.get('sent') else f"Failed to send email to {recipient}",
                "metadata": result
            }
        except Exception as e:
            logger.error(f"Email sending failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "type": "error",
                "message": f"Failed to send email: {str(e)}"
            }
    
    def _handle_library_upload(self, query: str, params: Dict) -> Dict[str, Any]:
        """Handle library document upload requests."""
        logger.info("📚 Handling library upload request...")
        logger.info(f"Query: {query}, Params: {params}")
        
        instructions = """<strong>How to Upload a Document to the Library</strong>

You can upload documents in two ways:

<strong>Option 1: Use the Library Tab (Easiest)</strong>
1. Click on the "Library" tab at the top of the page
2. Click "Choose File" and select your document (PDF, DOCX, or TXT)
3. Check "Auto-index to Milvus" if you want it indexed immediately
4. Click "Upload & Process Document"

<strong>Option 2: Upload via Chat (Below)</strong>
Use the file upload interface that appears below this message.

<strong>What happens after upload:</strong>
- File is saved and processed
- Text is extracted and cleaned
- Document is chunked into manageable pieces
- Processed data is saved as JSON
- (If auto-index enabled) Document is embedded and indexed into Milvus

The document will then be available for retrieval in the RAG system!"""
        
        result = {
            "success": True,
            "type": "library_upload",
            "message": instructions,
            "response": instructions,
            "show_upload_ui": True,  # Flag to show upload UI in chat
            "metadata": {
                "instruction_type": "file_upload",
                "upload_endpoint": "/api/upload-document"
            }
        }
        
        logger.info(f"Library upload handler returning: success={result['success']}, type={result['type']}, show_upload_ui={result.get('show_upload_ui')}")
        return result
    
    def _handle_awareness_email(self, query: str, params: Dict) -> Dict[str, Any]:
        """Handle awareness email batch sending."""
        logger.info("📧 Handling awareness email batch...")
        
        # CRITICAL: Clear any cached content - awareness emails should NOT use quiz/study guide content
        # This ensures awareness emails are independent tasks
        logger.info("Clearing cached content to ensure awareness email is independent")
        
        # Extract parameters
        topic = params.get("topic", "environmental sustainability")
        audience = params.get("audience", "general public")
        
        # Extract email addresses from query
        import re
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        recipients = re.findall(email_pattern, query)
        
        if not recipients:
            return {
                "success": False,
                "type": "error",
                "message": "Please provide email addresses. Example: 'Send awareness email about climate change to user1@example.com, user2@example.com'"
            }
        
        logger.info(f"Awareness email task - Topic: {topic}, Audience: {audience}, Recipients: {recipients}")
        
        # Generate email content directly (don't generate awareness content separately)
        try:
            from partd.email_tool import EmailTool
            import os
            
            email_tool = EmailTool(
                sender_email=os.getenv("SENDER_EMAIL", "sethharry123@gmail.com"),
                sender_password=os.getenv("SENDER_PASSWORD", "jkex gegg zrhm ryig"),
                llm_client=self.llm
            )
            
            # Generate email body using LLM reasoning
            logger.info(f"Generating awareness email content for topic: {topic}, audience: {audience}")
            email_body = email_tool.generate_outreach_email_with_reasoning(
                topic=topic,
                audience=audience
            )
            
            subject = f"Environmental Awareness: {topic.title()}"
            
            logger.info(f"Generated email body (length: {len(email_body)} chars). Sending to {len(recipients)} recipients...")
            logger.info(f"Recipients: {recipients}")
            logger.info(f"Subject: {subject}")
            
            # Send batch emails using EmailTools
            from parth.email_tools import EmailTools
            smtp_config = {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'sender_email': os.getenv("SENDER_EMAIL", "sethharry123@gmail.com"),
                'sender_password': os.getenv("SENDER_PASSWORD", "jkex gegg zrhm ryig")
            }
            logger.info(f"Initializing EmailTools with SMTP config...")
            email_tools = EmailTools(smtp_config)
            
            # Verify email tools is enabled
            if not hasattr(email_tools, 'email_available') or not email_tools.email_available:
                logger.error("Email tools not enabled. Check SMTP credentials.")
                logger.error(f"email_available: {getattr(email_tools, 'email_available', 'NOT SET')}")
                return {
                    "success": False,
                    "type": "error",
                    "message": "Email sending is not configured. Please check SMTP credentials."
                }
            
            logger.info(f"Email tools available: {email_tools.email_available}. Calling send_batch...")
            batch_result = email_tools.send_batch(
                recipients=recipients,
                subject=subject,
                body=email_body,
                max_retries=2
            )
            logger.info(f"send_batch completed. Result: {batch_result}")
            
            successful = batch_result.get('successful', 0)
            failed = batch_result.get('failed', 0)
            results = batch_result.get('results', [])
            
            logger.info(f"Email batch result: {successful} successful, {failed} failed")
            
            # Log detailed results
            for result in results:
                if result.get('success'):
                    logger.info(f"  ✓ Sent to {result.get('recipient')}")
                else:
                    logger.warning(f"  ✗ Failed to send to {result.get('recipient')}: {result.get('error', 'Unknown error')}")
            
            return {
                "success": successful > 0,
                "type": "awareness_email",
                "message": f"Mail sent to {successful} recipient(s)" if successful > 0 else f"Failed to send emails",
                "response": f"Mail sent to {successful} recipient(s): {', '.join([r.get('recipient', '') for r in results if r.get('success')])}" if successful > 0 else f"Failed to send emails. Check logs for details.",
                "metadata": {
                    "topic": topic,
                    "audience": audience,
                    "total_recipients": len(recipients),
                    "successful": successful,
                    "failed": failed,
                    "subject": subject,
                    "recipients": recipients,
                    "results": results
                }
            }
        
        except Exception as e:
            logger.error(f"Awareness email batch failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "type": "error",
                "message": f"Failed to send awareness emails: {str(e)}"
            }
    
    def _handle_document(self, query: str, params: Dict) -> Dict[str, Any]:
        """Handle document export."""
        logger.info("📄 Handling document export...")
        
        format_type = params.get("format", "pdf")
        
        # Get last generated content
        last_content = self.conversation.get_cached_context("last_generated_content")
        if not last_content:
            return {
                "success": False,
                "type": "error",
                "message": "No content available to export. Please generate content first."
            }
        
        # Use document export functionality
        try:
            from partd.export_tools import DocumentExporter
            exporter = DocumentExporter()
            
            if last_content.get("content_type") == "quiz":
                # Extract quiz data - it might be nested under "quiz" key
                quiz_data = last_content.get("quiz", last_content)
                if format_type == "pdf":
                    filepath = exporter.export_quiz_to_pdf(quiz_data)
                else:
                    filepath = exporter.export_quiz_to_docx(quiz_data)
            else:
                filepath = exporter.export_content_to_docx(
                    last_content,
                    last_content.get("content_type", "content")
                )
            
            # Cache filepath for email
            self.conversation.cache_context("last_exported_file", filepath)
            
            # Also update the cached content with filepath
            last_content = self.conversation.get_cached_context("last_generated_content")
            if last_content:
                last_content["filepath"] = filepath
                self.conversation.cache_context("last_generated_content", last_content)
            
            return {
                "success": True,
                "type": "document",
                "message": f"Document exported to {filepath}",
                "filepath": filepath,
                "format": format_type
            }
        except Exception as e:
            logger.error(f"Document export failed: {e}")
            return {
                "success": False,
                "type": "error",
                "message": f"Failed to export document: {str(e)}"
            }
    
    def _handle_combined(self, query: str, params: Dict, task_info: Dict) -> Dict[str, Any]:
        """Handle combined tasks (e.g., generate and send)."""
        logger.info("🔄 Handling combined task...")
        
        subtasks = task_info.get("subtasks", [])
        
        # If email is detected but document isn't, and format is mentioned, add document export
        has_email = any(st.get("task") == "email" for st in subtasks)
        has_document = any(st.get("task") == "document" for st in subtasks)
        has_generation = any(st.get("task") in ["quiz", "study_guide", "awareness"] for st in subtasks)
        
        if has_email and not has_document and has_generation:
            # Check if format is mentioned in email params
            email_params = params.get("email", {})
            if "format" in email_params or "pdf" in query.lower() or "docx" in query.lower():
                # Add document export subtask
                subtasks.append({
                    "task": "document",
                    "handler": "document_handler",
                    "confidence": "high"
                })
                # Extract format for document
                if "document" not in params:
                    params["document"] = {}
                if "format" not in params["document"]:
                    query_lower = query.lower()
                    if "pdf" in query_lower:
                        params["document"]["format"] = "pdf"
                    elif "docx" in query_lower or "word" in query_lower:
                        params["document"]["format"] = "docx"
                    else:
                        params["document"]["format"] = "pdf"  # Default
        
        # Sort subtasks to ensure correct order: generation -> export -> email
        task_priority = {
            "quiz": 1,
            "study_guide": 1,
            "awareness": 1,
            "document": 2,
            "email": 3
        }
        subtasks.sort(key=lambda x: task_priority.get(x.get("task", ""), 99))
        
        results = []
        
        # Execute subtasks in order
        for subtask in subtasks:
            task_type = subtask.get("task")
            handler = subtask.get("handler")
            
            # Get parameters for this subtask
            task_params = params.get(task_type, {})
            
            try:
                if handler == "quiz_handler":
                    result = self._handle_quiz(query, task_params)
                elif handler == "study_guide_handler":
                    result = self._handle_study_guide(query, task_params)
                elif handler == "awareness_handler":
                    result = self._handle_awareness(query, task_params)
                elif handler == "document_handler":
                    # If format not in task_params, check if it's in email params
                    if "format" not in task_params:
                        email_params = params.get("email", {})
                        if "format" in email_params:
                            task_params["format"] = email_params["format"]
                    result = self._handle_document(query, task_params)
                elif handler == "email_handler":
                    result = self._handle_email(query, task_params)
                else:
                    continue
                
                results.append(result)
                
                # Cache content for next steps
                if result.get("success"):
                    if "quiz" in result:
                        self.conversation.cache_context("last_generated_content", {
                            "content_type": "quiz",
                            "topic": result["quiz"].get("topic", ""),
                            "quiz": result["quiz"]
                        })
                    elif "study_guide" in result:
                        self.conversation.cache_context("last_generated_content", {
                            "content_type": "study_guide",
                            "topic": result["study_guide"].get("topic", ""),
                            "content": result["study_guide"]
                        })
            
            except Exception as e:
                logger.error(f"Subtask {task_type} failed: {e}")
                results.append({
                    "success": False,
                    "task": task_type,
                    "error": str(e)
                })
        
        # Combine results
        success_count = sum(1 for r in results if r.get("success"))
        
        # Extract quiz/study_guide/awareness content from results for UI display
        quiz_data = None
        study_guide_data = None
        awareness_data = None
        email_result = None
        
        for result in results:
            if result.get("success"):
                if "quiz" in result:
                    quiz_data = result["quiz"]
                elif "study_guide" in result:
                    study_guide_data = result["study_guide"]
                elif "awareness_content" in result:
                    awareness_data = result["awareness_content"]
                elif result.get("type") == "email":
                    email_result = result
        
        # Create user-friendly message
        if email_result and email_result.get("success"):
            # If email was sent, use a friendly message
            recipient = email_result.get("metadata", {}).get("recipient") or params.get("email", {}).get("recipient", "your email")
            message = f"Mail sent to {recipient}"
        else:
            message = f"Completed {success_count}/{len(results)} tasks"
        
        response = {
            "success": success_count > 0,
            "type": "combined",
            "message": message,
            "response": message,  # Also set response for UI
            "results": results,
            "metadata": {
                "num_subtasks": len(subtasks),
                "completed": success_count
            }
        }
        
        # Add content data for UI display
        if quiz_data:
            response["quiz"] = quiz_data
            if not email_result:
                response["response"] = f"Generated quiz on {quiz_data.get('topic', 'topic')} with {quiz_data.get('num_questions', 0)} questions"
        elif study_guide_data:
            response["study_guide"] = study_guide_data
            if not email_result:
                response["response"] = f"Generated study guide on {study_guide_data.get('topic', 'topic')}"
        elif awareness_data:
            response["awareness_content"] = awareness_data
            if not email_result:
                response["response"] = f"Generated {awareness_data.get('format', 'content')} on {awareness_data.get('topic', 'topic')}"
        
        return response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



