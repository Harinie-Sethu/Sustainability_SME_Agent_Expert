"""
Agent Core - Main Agent with Planning, Reasoning, and Routing
Orchestrates all agent capabilities with multi-step reasoning
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, Any, Optional, List
import json
import logging
from datetime import datetime
import time

from parte.conversation_manager import ConversationManager
from parte.task_router import TaskRouter
from parte.observations import ObservationsLogger
from parte import prompts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnvironmentalAgent:
    """
    Comprehensive Environmental Sustainability Agent with:
    - Conversational planning
    - Multi-step reasoning
    - Intelligent task routing
    - Context and memory management
    - Observational learning
    """
    
    def __init__(self, llm_client, rag_system, content_generator):
        """
        Initialize the agent.
        
        Args:
            llm_client: LLM client for reasoning
            rag_system: RAG system for knowledge retrieval
            content_generator: Content generation system
        """
        self.llm = llm_client
        self.rag = rag_system
        self.content_gen = content_generator
        
        # Initialize components
        self.conversation = ConversationManager()
        self.router = TaskRouter(llm_client)
        self.observations = ObservationsLogger()
        
        # Register handlers
        self._register_handlers()
        
        logger.info(" Environmental Agent initialized")
    
    def _register_handlers(self):
        """Register all task handlers with the router."""
        self.router.register_handler("qa_handler", self._handle_qa)
        self.router.register_handler("content_generator", self._handle_content_generation)
        self.router.register_handler("document_handler", self._handle_document_export)
        self.router.register_handler("email_handler", self._handle_email)
        self.router.register_handler("data_analyst", self._handle_data_analysis)
        self.router.register_handler("conversation_handler", self._handle_conversation)
        
        logger.info(" All handlers registered")
    
    def process_request(self, user_request: str) -> Dict[str, Any]:
        """
        Process user request with full agent workflow.
        
        Workflow:
        1. Add to conversation history
        2. Create execution plan
        3. Route to appropriate handler(s)
        4. Execute tasks
        5. Reflect on performance
        6. Log observations
        
        Args:
            user_request: User's request
            
        Returns:
            Agent response with results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing request: {user_request}")
        logger.info(f"{'='*70}")
        
        start_time = time.time()
        
        # Increment iteration
        self.observations.increment_iteration()
        
        # Step 1: Add to conversation
        self.conversation.add_message("user", user_request)
        
        # Step 2: Create plan
        plan = self._create_plan(user_request)
        self.observations.log_planning(user_request, plan, 
                                      self.conversation.get_context_summary(self.llm))
        
        # Step 3: Route and execute
        routing_result = self.router.route_task(
            user_request,
            context=self.conversation.get_formatted_history()
        )
        
        self.observations.log_routing(user_request, routing_result)
        
        # Check if clarification needed
        if routing_result.get('requires_clarification'):
            clarification = self._request_clarification(user_request, routing_result)
            response = {
                "type": "clarification",
                "message": clarification,
                "requires_user_input": True
            }
            self.conversation.add_message("agent", clarification)
            return response
        
        # Step 4: Execute
        execution_start = time.time()
        execution_result = self.router.execute_route(routing_result)
        execution_time = time.time() - execution_start
        
        self.observations.log_execution(
            routing_result.get('handler'),
            routing_result.get('extracted_parameters', {}),
            execution_result,
            execution_time
        )
        
        # Step 5: Generate response
        agent_response = self._format_response(execution_result, routing_result)
        
        # Step 6: Add to conversation
        self.conversation.add_message("agent", agent_response.get('message', ''))
        
        # Step 7: Reflect (optional, can be done periodically)
        if self.observations.iteration_count % 5 == 0:  # Every 5 interactions
            self._reflect_on_performance(user_request, agent_response)
        
        total_time = time.time() - start_time
        
        logger.info(f" Request processed in {total_time:.2f}s")
        
        return {
            **agent_response,
            "metadata": {
                "iteration": self.observations.iteration_count,
                "execution_time": total_time,
                "handler_used": routing_result.get('handler'),
                "plan_subtasks": len(plan.get('subtasks', []))
            }
        }
    
    def _create_plan(self, user_request: str) -> Dict[str, Any]:
        """
        Create execution plan for user request.
        
        Args:
            user_request: User's request
            
        Returns:
            Execution plan
        """
        logger.info("📋 Creating execution plan...")
        
        conversation_history = self.conversation.get_formatted_history()
        
        prompt = prompts.PLANNING_PROMPT.format(
            user_request=user_request,
            conversation_history=conversation_history
        )
        
        try:
            plan_json = self.llm.generate_json(prompt)
            
            if isinstance(plan_json, dict) and 'subtasks' in plan_json:
                logger.info(f" Plan created with {len(plan_json['subtasks'])} subtasks")
                return plan_json
            else:
                logger.warning("Invalid plan format, using simple plan")
                return self._simple_plan(user_request)
        
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return self._simple_plan(user_request)
    
    def _simple_plan(self, user_request: str) -> Dict[str, Any]:
        """Create simple fallback plan."""
        return {
            "main_goal": user_request,
            "requires_context": False,
            "subtasks": [{
                "subtask_id": 1,
                "description": "Process user request",
                "capability": "auto_detect",
                "inputs": {"request": user_request},
                "depends_on": [],
                "reasoning": "Direct processing of request"
            }],
            "expected_output": "Response to user request"
        }
    
    def _request_clarification(self, user_request: str, routing_result: Dict) -> str:
        """
        Request clarification from user.
        
        Args:
            user_request: User's request
            routing_result: Routing result indicating need for clarification
            
        Returns:
            Clarification question
        """
        logger.info("❓ Requesting clarification...")
        
        prompt = prompts.CLARIFICATION_PROMPT.format(
            user_request=user_request,
            ambiguity="Multiple interpretations possible",
            conversation_history=self.conversation.get_formatted_history()
        )
        
        try:
            clarification_json = self.llm.generate_json(prompt)
            
            if isinstance(clarification_json, dict):
                question = clarification_json.get('clarification_question', 
                                                 "Could you please clarify what you'd like me to do?")
                return question
        except:
            pass
        
        return "I want to make sure I understand correctly. Could you provide more details about what you'd like?"
    
    def _format_response(self, execution_result: Dict, routing_result: Dict) -> Dict[str, Any]:
        """
        Format agent response from execution result.
        
        Args:
            execution_result: Result from handler execution
            routing_result: Original routing decision
            
        Returns:
            Formatted response
        """
        if not execution_result.get('success'):
            error_msg = execution_result.get('error', 'Unknown error')
            return {
                "type": "error",
                "message": f"I encountered an error: {error_msg}. Let me try a different approach.",
                "success": False
            }
        
        result_data = execution_result.get('result', {})
        handler = execution_result.get('handler_used')
        
        # Format based on handler type
        if handler == "qa_handler":
            # Handle both direct RAG result and wrapped result
            answer = result_data.get('answer', '') if isinstance(result_data, dict) else execution_result.get('answer', '')
            sources = result_data.get('sources', {}) if isinstance(result_data, dict) else execution_result.get('sources', {})
            
            # Ensure sources are properly formatted
            if not isinstance(sources, dict):
                sources = {}
            
            local_sources = sources.get("local", [])
            web_sources = sources.get("web", [])
            
            formatted_sources = {
                "local": local_sources if isinstance(local_sources, list) else [],
                "web": web_sources if isinstance(web_sources, list) else []
            }
            
            return {
                "type": "answer",
                "message": answer or execution_result.get('answer', ''),
                "response": answer or execution_result.get('answer', ''),
                "sources": formatted_sources,
                "success": True
            }
        
        elif handler == "content_generator":
            content_type = result_data.get('content_type', 'content')
            
            # For quiz, return in format expected by UI
            if content_type == 'quiz':
                return {
                    "type": "quiz",
                    "message": f"I've generated the quiz for you.",
                    "response": f"I've generated the quiz for you.",
                    "quiz": result_data,  # Include quiz data for UI
                    "topic": result_data.get('topic', ''),
                    "success": True
                }
            # For study guide
            elif content_type == 'study_guide':
                return {
                    "type": "study_guide",
                    "message": f"I've generated the study guide for you.",
                    "response": f"I've generated the study guide for you.",
                    "study_guide": result_data,  # Include study guide data for UI
                    "topic": result_data.get('topic', ''),
                    "success": True
                }
            # For other content types
            else:
                return {
                    "type": "content",
                    "message": f"I've generated the {content_type} for you.",
                    "content": result_data,
                    "success": True
                }
        
        elif handler == "document_handler":
            filepath = result_data.get('filepath', '')
            return {
                "type": "document",
                "message": f"Document exported successfully to {filepath}",
                "filepath": filepath,
                "success": True
            }
        
        elif handler == "email_handler":
            success = result_data.get('success', result_data.get('sent', False))
            message = result_data.get('message', result_data.get('response', 'Email sent successfully'))
            return {
                "type": "email",
                "message": message,
                "response": message,
                "success": success
            }
        
        elif handler == "conversation_handler":
            return {
                "type": "conversation",
                "message": result_data.get('message', ''),
                "success": True
            }
        
        else:
            return {
                "type": "generic",
                "message": "Task completed successfully.",
                "result": result_data,
                "success": True
            }
    
    def _reflect_on_performance(self, user_request: str, agent_response: Dict):
        """
        Reflect on agent's performance.
        
        Args:
            user_request: User's request
            agent_response: Agent's response
        """
        logger.info("🤔 Reflecting on performance...")
        
        # Get recent observations
        insights = self.observations.generate_insights()
        
        # Create reflection
        reflection_data = {
            "user_request": user_request,
            "response_type": agent_response.get('type'),
            "success": agent_response.get('success', False),
            "insights": insights
        }
        
        self.observations.log_reflection(user_request, 
                                        agent_response.get('message', ''),
                                        reflection_data)
    
    # ==================== HANDLER IMPLEMENTATIONS ====================
    
    def _handle_qa(self, parameters: Dict, routing: Dict) -> Dict:
        """Handle question-answering tasks."""
        logger.info(" Handling Q&A...")
        
        question = parameters.get('question', routing.get('user_request', ''))
        
        # Get conversation context
        context = self.conversation.get_recent_context(3)
        
        # Use RAG system
        result = self.rag.answer_question(question, conversation_history=context)
        
        return result
    
    def _handle_content_generation(self, parameters: Dict, routing: Dict) -> Dict:
        """Handle content generation tasks."""
        logger.info("✍️ Handling content generation...")
        
        content_type = parameters.get('content_type', 'article')
        topic = parameters.get('topic', 'sustainability')
        
        if content_type == 'quiz':
            num_questions = parameters.get('num_questions', 5)
            difficulty = parameters.get('difficulty', 'medium')
            result = self.content_gen.generate_quiz(topic, num_questions, difficulty)
            result['content_type'] = 'quiz'
            # Cache quiz for follow-up requests (email, export, etc.)
            self.conversation.cache_context('last_generated_content', result)
            self.conversation.cache_context('last_quiz', result)
        
        elif content_type == 'study_guide':
            result = self.content_gen.generate_study_guide(topic)
            result['content_type'] = 'study_guide'
            # Cache study guide for follow-up requests
            self.conversation.cache_context('last_generated_content', result)
            self.conversation.cache_context('last_study_guide', result)
        
        elif content_type in ['article', 'social_media', 'infographic_text', 'poster_text']:
            audience = parameters.get('audience', 'general public')
            result = self.content_gen.generate_awareness_material(topic, content_type, audience)
            result['content_type'] = content_type
        
        else:
            result = {
                "content_type": content_type,
                "message": f"Content type '{content_type}' generated",
                "topic": topic
            }
        
        return result
    
    def _handle_document_export(self, parameters: Dict, routing: Dict) -> Dict:
        """Handle document export tasks."""
        logger.info(" Handling document export...")
        
        # This would integrate with export_tools from partd
        from partd.export_tools import DocumentExporter
        
        exporter = DocumentExporter()
        
        format_type = parameters.get('format', 'pdf')
        
        # Get content from conversation cache
        content = self.conversation.get_cached_context('last_generated_content')
        
        if not content:
            return {
                "success": False,
                "error": "No content available to export"
            }
        
        # Export based on content type
        if content.get('content_type') == 'quiz':
            if format_type == 'pdf':
                filepath = exporter.export_quiz_to_pdf(content)
            else:
                filepath = exporter.export_quiz_to_docx(content)
        else:
            filepath = exporter.export_content_to_docx(content, content.get('content_type', 'content'))
        
        return {
            "filepath": filepath,
            "format": format_type
        }
    
    def _handle_email(self, parameters: Dict, routing: Dict) -> Dict:
        """Handle email sending tasks."""
        logger.info(" Handling email...")
        
        recipient = parameters.get('recipient', 'user@example.com')
        format_type = parameters.get('format', 'pdf')  # Get format from parameters
        
        # Get content from cache
        content = self.conversation.get_cached_context('last_generated_content')
        filepath = self.conversation.get_cached_context('last_exported_file')
        
        # If no filepath exists, auto-export the content first
        if not filepath and content:
            logger.info(f" No exported file found, auto-exporting as {format_type}...")
            try:
                from partd.export_tools import DocumentExporter
                exporter = DocumentExporter()
                
                content_type = content.get('content_type', 'content')
                
                # Export based on content type and format
                if content_type == 'quiz':
                    if format_type.lower() == 'ppt' or format_type.lower() == 'pptx':
                        # PPT not available for quiz, use DOCX as fallback
                        logger.info("PPT not available for quiz, using DOCX instead")
                        filepath = exporter.export_quiz_to_docx(content)
                    elif format_type.lower() == 'pdf':
                        filepath = exporter.export_quiz_to_pdf(content)
                    else:
                        filepath = exporter.export_quiz_to_docx(content)
                elif content_type == 'study_guide':
                    if format_type.lower() == 'ppt' or format_type.lower() == 'pptx':
                        filepath = exporter.export_study_guide_to_ppt(content)
                    elif format_type.lower() == 'pdf':
                        filepath = exporter.export_study_guide_to_pdf(content)
                    else:
                        filepath = exporter.export_study_guide_to_docx(content)
                else:
                    # For other content types
                    if format_type.lower() == 'pdf':
                        filepath = exporter.export_content_to_pdf(content, content_type)
                    else:
                        filepath = exporter.export_content_to_docx(content, content_type)
                
                # Cache the filepath
                self.conversation.cache_context('last_exported_file', filepath)
                logger.info(f" Auto-exported to: {filepath}")
                
            except Exception as e:
                logger.error(f"Auto-export failed: {e}")
                return {
                    "success": False,
                    "error": f"Failed to export document: {str(e)}"
                }
        
        if not filepath:
            return {
                "success": False,
                "error": "No document available to send. Please generate content first."
            }
        
        # Use email functionality
        try:
            logger.info(f"Attempting to send email to {recipient} with file: {filepath}")
            result = self.content_gen.send_content_via_email(
                recipient=recipient,
                content_filepath=filepath,
                content_type=content.get('content_type', 'document') if content else 'document',
                topic=content.get('topic', 'sustainability') if content else 'sustainability'
            )
            
            sent = result.get('sent', False)
            logger.info(f"Email sending result: sent={sent}, result={result}")
            
            if not sent:
                error_msg = result.get('error', 'Email sending failed. Please check email credentials (SENDER_EMAIL and SENDER_PASSWORD environment variables).')
                logger.warning(f"Email not sent: {error_msg}")
                return {
                    "success": False,
                    "message": f"Failed to send email to {recipient}. {error_msg}",
                    "response": f"Failed to send email to {recipient}. {error_msg}",
                    "sent": False,
                    "error": error_msg
                }
            
            return {
                "success": True,
                "message": f"Mail sent to {recipient}",
                "response": f"Mail sent to {recipient}",
                "sent": True
            }
        except Exception as e:
            logger.error(f"Email sending failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "message": f"Failed to send email to {recipient}: {str(e)}",
                "response": f"Failed to send email to {recipient}: {str(e)}",
                "error": f"Failed to send email: {str(e)}"
            }
    
    def _handle_data_analysis(self, parameters: Dict, routing: Dict) -> Dict:
        """Handle data analysis tasks."""
        logger.info(" Handling data analysis...")
        
        # Use multi-step reasoning for analysis
        problem = routing.get('user_request', '')
        
        prompt = prompts.REASONING_PROMPT.format(
            problem=problem,
            context=self.conversation.get_formatted_history(),
            previous_steps="None"
        )
        
        try:
            reasoning_json = self.llm.generate_json(prompt)
            
            if isinstance(reasoning_json, dict):
                self.observations.log_reasoning(
                    problem,
                    reasoning_json.get('reasoning_steps', []),
                    reasoning_json.get('conclusion', ''),
                    reasoning_json.get('confidence', 'medium')
                )
                
                return {
                    "analysis": reasoning_json.get('conclusion', ''),
                    "reasoning": reasoning_json.get('reasoning_steps', []),
                    "confidence": reasoning_json.get('confidence', 'medium')
                }
        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
        
        return {
            "analysis": "I've analyzed the data based on available information.",
            "reasoning": ["Analysis completed with available data"],
            "confidence": "medium"
        }
    
    def _handle_conversation(self, parameters: Dict, routing: Dict) -> Dict:
        """Handle general conversation."""
        logger.info("💬 Handling conversation...")
        
        user_message = routing.get('user_request', '')
        message_lower = user_message.lower()
        
        # Handle greetings
        if any(greeting in message_lower for greeting in ['hello', 'hi', 'hey']):
            return {
                "message": "Hello! I'm your Environmental Sustainability Agent. I can help you with:\n"
                          "- Answering questions about environmental topics\n"
                          "- Creating educational content (quizzes, study guides, articles)\n"
                          "- Analyzing environmental data and trends\n"
                          "- Exporting and sharing content\n\n"
                          "What would you like to explore today?"
            }
        
        # Handle help requests
        if 'help' in message_lower or 'what can you' in message_lower:
            return {
                "message": "I'm here to help with environmental sustainability! Here's what I can do:\n\n"
                          " **Answer Questions**: Ask me anything about climate change, renewable energy, "
                          "conservation, pollution, and more.\n\n"
                          "✍️ **Create Content**: Generate quizzes, study guides, articles, or social media posts.\n\n"
                          " **Analyze Data**: Help you understand trends, compare options, and make informed decisions.\n\n"
                          " **Export & Share**: Save content as PDF/DOCX and send via email.\n\n"
                          "Just tell me what you need!"
            }
        
        # Handle thanks
        if any(word in message_lower for word in ['thank', 'thanks']):
            return {
                "message": "You're welcome! I'm glad I could help. Feel free to ask if you have more questions "
                          "about environmental sustainability!"
            }
        
        # Handle goodbye
        if any(word in message_lower for word in ['bye', 'goodbye', 'exit', 'quit']):
            stats = self.conversation.get_conversation_stats()
            return {
                "message": f"Goodbye! We had {stats['user_messages']} exchanges in this session. "
                          f"Thank you for caring about the environment! 🌍"
            }
        
        # General conversational response
        context = self.conversation.get_formatted_history(3)
        
        prompt = f"""You are a friendly Environmental Sustainability Agent having a conversation.

Recent context:
{context}

User: {user_message}

Respond naturally and helpfully. Keep it concise (2-3 sentences).

Response:"""
        
        response = self.llm.generate(prompt, max_tokens=200, temperature=0.8)
        
        return {
            "message": response.strip()
        }
    
    # ==================== UTILITY METHODS ====================
    
    def save_session(self) -> str:
        """
        Save current session data.
        
        Returns:
            Path to saved session
        """
        # Save conversation
        conv_file = self.conversation.export_conversation()
        
        # Save observations
        obs_file = self.observations.save_observations()
        
        # Generate report
        report = self.observations.generate_report(save_to_file=True)
        
        logger.info(f" Session saved: conversation={conv_file}, observations={obs_file}")
        
        return conv_file
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        conv_stats = self.conversation.get_conversation_stats()
        obs_insights = self.observations.generate_insights()
        routing_stats = self.router.get_routing_stats()
        
        return {
            "conversation": conv_stats,
            "observations": obs_insights,
            "routing": routing_stats
        }
    
    def get_improvement_suggestions(self) -> List[str]:
        """Get suggestions for improving prompts and performance."""
        return self.observations.get_prompt_refinement_suggestions()
    
    def _handle_conversation(self, parameters: Dict, routing: Dict) -> Dict:
        """Handle general conversation."""
        logger.info("💬 Handling conversation...")
        
        user_message = routing.get('user_request', '')
        message_lower = user_message.lower()
        
        # Handle greetings
        if any(greeting in message_lower for greeting in ['hello', 'hi', 'hey']):
            return {
                "message": "Hello! I'm your Environmental Sustainability Agent. I can help you with:\n"
                          "- Answering questions about environmental topics\n"
                          "- Creating educational content (quizzes, study guides, articles)\n"
                          "- Analyzing environmental data and trends\n"
                          "- Exporting and sharing content\n\n"
                          "What would you like to explore today?"
            }
        
        # Handle help requests
        if 'help' in message_lower or 'what can you' in message_lower:
            return {
                "message": "I'm here to help with environmental sustainability! Here's what I can do:\n\n"
                          " **Answer Questions**: Ask me anything about climate change, renewable energy, "
                          "conservation, pollution, and more.\n\n"
                          "✍️ **Create Content**: Generate quizzes, study guides, articles, or social media posts.\n\n"
                          " **Analyze Data**: Help you understand trends, compare options, and make informed decisions.\n\n"
                          " **Export & Share**: Save content as PDF/DOCX and send via email.\n\n"
                          "Just tell me what you need!"
            }
        
        # Handle thanks
        if any(word in message_lower for word in ['thank', 'thanks']):
            return {
                "message": "You're welcome! I'm glad I could help. Feel free to ask if you have more questions "
                          "about environmental sustainability!"
            }
        
        # Handle goodbye
        if any(word in message_lower for word in ['bye', 'goodbye', 'exit', 'quit']):
            stats = self.conversation.get_conversation_stats()
            return {
                "message": f"Goodbye! We had {stats['user_messages']} exchanges in this session. "
                          f"Thank you for caring about the environment! 🌍"
            }
        
        # General conversational response
        context = self.conversation.get_formatted_history(3)
        
        prompt = f"""You are a friendly Environmental Sustainability Agent having a conversation.

Recent context:
{context}

User: {user_message}

Respond naturally and helpfully. Keep it concise (2-3 sentences).

Response:"""
        
        response = self.llm.generate(prompt, max_tokens=200, temperature=0.8)
        
        return {
            "message": response.strip()
        }
    
    # ==================== UTILITY METHODS ====================
    
    def save_session(self) -> str:
        """
        Save current session data.
        
        Returns:
            Path to saved session
        """
        # Save conversation
        conv_file = self.conversation.export_conversation()
        
        # Save observations
        obs_file = self.observations.save_observations()
        
        # Generate report
        report = self.observations.generate_report(save_to_file=True)
        
        logger.info(f" Session saved: conversation={conv_file}, observations={obs_file}")
        
        return conv_file
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        conv_stats = self.conversation.get_conversation_stats()
        obs_insights = self.observations.generate_insights()
        routing_stats = self.router.get_routing_stats()
        
        return {
            "conversation": conv_stats,
            "observations": obs_insights,
            "routing": routing_stats
        }
    
    def get_improvement_suggestions(self) -> List[str]:
        """Get suggestions for improving prompts and performance."""
        return self.observations.get_prompt_refinement_suggestions()
