"""
Task Handlers for Core SME Capabilities
Task 1: Expert Q&A
Task 2: Educational Content Generation (Quiz, Study Guides, Awareness)
Includes multi-step reasoning and self-learning
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, List, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentGenerator:
    """
    Content generator implementing both required tasks with:
    - Multi-step reasoning
    - RAG integration
    - Self-learning capabilities (BONUS)
    """
    
    def __init__(self, rag_system, llm_client):
        """
        Initialize content generator.
        
        Args:
            rag_system: Enhanced RAG system
            llm_client: Gemini LLM client
        """
        self.rag = rag_system
        self.llm = llm_client
        
        logger.info("✓ Content generator initialized")
    
    # ==================== TASK 1: EXPERT Q&A ====================
    
    def answer_question(self, question: str, 
                       conversation_history: Optional[List[Dict]] = None) -> Dict:
        """
        TASK 1: Answer questions using RAG with adaptive explanations.
        
        Features:
        - RAG-based retrieval
        - Web search fallback
        - Multi-step reasoning
        - Conversational context
        - Self-learning from feedback
        
        Args:
            question: User's question
            conversation_history: Previous conversation messages
            
        Returns:
            Dict with answer and metadata
        """
        return self.rag.answer_question(question, conversation_history=conversation_history)
    
    # ==================== TASK 2: QUIZ GENERATION ====================
    
    def generate_quiz(self, topic: str, num_questions: int = 5, 
                     difficulty: str = "medium") -> Dict:
        """
        TASK 2: Generate educational quiz with multi-step reasoning.
        
        Multi-step process:
        1. Gather information from RAG
        2. Analyze topic and identify key concepts
        3. Generate questions with reasoning
        4. Create detailed explanations
        
        Args:
            topic: Quiz topic
            num_questions: Number of questions
            difficulty: easy/medium/hard
            
        Returns:
            Dict with quiz data
        """
        logger.info(f"Generating quiz: {topic} ({num_questions} questions, {difficulty})")
        
        # Step 1: Gather information from RAG
        info_query = f"Provide comprehensive educational information about {topic} for creating a quiz"
        info_result = self.rag.answer_question(info_query)
        context = info_result['answer']
        
        # Step 2: Multi-step chain generation
        chain_steps = [
            {
                "name": "analyze_topic",
                "prompt": f"""Analyze the topic "{topic}" for quiz generation.

Context: {context}

Identify:
1. Key concepts that should be covered (focus on: {topic})
2. Appropriate complexity for {difficulty} difficulty
3. Important facts and principles related to {topic}

IMPORTANT: Stay focused on the topic "{topic}". Do not deviate to unrelated topics.

Return JSON:
{{
  "key_concepts": ["concept1 related to {topic}", "concept2 related to {topic}", ...],
  "complexity_level": "description for {difficulty} level",
  "important_facts": ["fact1 about {topic}", "fact2 about {topic}", ...]
}}""",
                "temperature": 0.5,
                "output_key": "analysis"
            },
            {
                "name": "generate_questions",
                "prompt": f"""Generate EXACTLY {num_questions} multiple-choice questions about {topic}.

Analysis: {{analysis}}
Difficulty: {difficulty}
Context: {context[:1000]}

CRITICAL REQUIREMENTS:
- Generate EXACTLY {num_questions} questions (not more, not less)
- All questions must be about: {topic}
- Difficulty level: {difficulty}
- Each question must have exactly 4 options (A, B, C, D)
- Each question must have one clearly correct answer
- Use proper spelling and grammar
- Questions should be educational and relevant to the topic
- For {difficulty} difficulty: {"use complex concepts and require deep understanding" if difficulty == "hard" else "use moderate complexity" if difficulty == "medium" else "use basic concepts"}

Return JSON array with EXACTLY {num_questions} items:
[
  {{
    "question": "Clear, well-written question about {topic}?",
    "options": {{
      "A": "Option A text",
      "B": "Option B text",
      "C": "Option C text",
      "D": "Option D text"
    }},
    "correct_answer": "A",
    "explanation": "Clear explanation of why the answer is correct"
  }}
]""",
                "temperature": 0.6,
                "output_key": "questions_raw"
            },
            {
                "name": "enhance_explanations",
                "prompt": f"""Enhance the explanations with step-by-step reasoning.

Questions: {{questions_raw}}
Topic: {topic}

For each explanation:
1. Explain why the correct answer is right
2. Explain why other options are wrong
3. Provide environmental context
4. Use clear, educational language

Return the same JSON with enhanced explanations.""",
                "temperature": 0.7,
                "output_key": "final_quiz"
            }
        ]
        
        # Execute chain
        try:
            chain_result = self.llm.chain_generate(chain_steps)
            
            # Parse final quiz
            final_quiz_text = chain_result.get('final_quiz', '')
            questions = self._parse_quiz_json(final_quiz_text, difficulty)
            
            if not questions:
                # Try parsing from questions_raw
                questions_raw = chain_result.get('questions_raw', '')
                questions = self._parse_quiz_json(questions_raw, difficulty)
            
            if not questions:
                logger.warning("Chain generation failed, using fallback")
                return self._generate_quiz_fallback(topic, num_questions, difficulty, context)
            
            # Validate number of questions
            if len(questions) != num_questions:
                logger.warning(f"Generated {len(questions)} questions but requested {num_questions}")
                # Trim or pad as needed
                if len(questions) > num_questions:
                    questions = questions[:num_questions]
                elif len(questions) < num_questions:
                    logger.warning(f"Only {len(questions)} questions generated, less than requested {num_questions}")
            
            # Ensure topic is preserved correctly
            final_topic = topic.strip()
            
            return {
                "topic": final_topic,
                "num_questions": len(questions),
                "difficulty": difficulty,
                "questions": questions,
                "metadata": {
                    "used_chain_reasoning": True,
                    "used_web_search": info_result['metadata'].get('used_web_search', False),
                    "rag_confidence": info_result['metadata'].get('retrieval_confidence', 0),
                    "requested_questions": num_questions,
                    "generated_questions": len(questions)
                }
            }
        
        except Exception as e:
            logger.error(f"Quiz chain generation failed: {e}")
            return self._generate_quiz_fallback(topic, num_questions, difficulty, context)
    
    def _parse_quiz_json(self, text: str, difficulty: str = "medium") -> List[Dict]:
        """Parse quiz JSON from text."""
        try:
            # Find JSON in text
            start = text.find('[')
            end = text.rfind(']') + 1
            
            if start != -1 and end > start:
                json_text = text[start:end]
                questions = json.loads(json_text)
                return self._normalize_quiz_format(questions, difficulty)
            
            # Try parsing as dict with questions key
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                json_text = text[start:end]
                data = json.loads(json_text)
                if 'questions' in data:
                    return self._normalize_quiz_format(data['questions'], difficulty)
            
            return []
        
        except json.JSONDecodeError:
            return []
    
    def _normalize_quiz_format(self, questions: List[Dict], difficulty: str = "medium") -> List[Dict]:
        """Normalize quiz format."""
        normalized = []
        
        for q in questions:
            # Handle options format
            if "options" in q:
                if isinstance(q["options"], dict):
                    options = q["options"]
                elif isinstance(q["options"], list):
                    options = {}
                    for i, opt in enumerate(q["options"]):
                        letter = chr(65 + i)
                        opt_text = opt.split(". ", 1)[-1] if ". " in opt else opt
                        options[letter] = opt_text
                else:
                    options = {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}
            else:
                options = {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}
            
            # Handle answer format
            answer = q.get("answer", q.get("correct_answer", "A"))
            if len(answer) > 1:
                answer = answer[0].upper()
            
            normalized.append({
                "question": q.get("question", "Question pending"),
                "options": options,
                "correct_answer": answer,
                "explanation": q.get("explanation", "Explanation pending"),
                "difficulty": q.get('difficulty', difficulty)
            })
        
        return normalized
    
    def _generate_quiz_fallback(self, topic: str, num_questions: int, 
                                difficulty: str, context: str) -> Dict:
        """Fallback quiz generation without chain."""
        prompt = f"""Create {num_questions} {difficulty} multiple-choice questions about {topic}.

Context: {context[:1500]}

Return JSON array:
[
  {{
    "question": "question?",
    "options": {{"A": "opt1", "B": "opt2", "C": "opt3", "D": "opt4"}},
    "correct_answer": "A",
    "explanation": "explanation"
  }}
]"""
        
        quiz_data = self.llm.generate_json(prompt)
        
        if isinstance(quiz_data, list):
            questions = quiz_data
        elif isinstance(quiz_data, dict) and 'questions' in quiz_data:
            questions = quiz_data['questions']
        else:
            questions = []
        
        return {
            "topic": topic,
            "num_questions": len(questions),
            "difficulty": difficulty,
            "questions": self._normalize_quiz_format(questions, difficulty) if questions else self._create_dummy_quiz(topic, num_questions, difficulty),
            "metadata": {
                "used_chain_reasoning": False,
                "fallback": True
            }
        }
    
    def _create_dummy_quiz(self, topic: str, num_questions: int, difficulty: str) -> List[Dict]:
        """Create dummy quiz as ultimate fallback."""
        return [{
            "question": f"What is an important aspect of {topic} related to environmental sustainability?",
            "options": {
                "A": "It has no environmental impact",
                "B": "It requires sustainable practices and careful consideration",
                "C": "It should be completely avoided",
                "D": "It is unrelated to environmental concerns"
            },
            "correct_answer": "B",
            "explanation": f"{topic} is an important environmental topic that requires sustainable practices.",
            "difficulty": difficulty
        }] * min(num_questions, 1)
    
    # ==================== TASK 2: STUDY GUIDE GENERATION ====================
    
    def generate_study_guide(self, topic: str) -> Dict:
        """
        TASK 2: Generate comprehensive study guide.
        
        Multi-step process:
        1. Gather information from RAG
        2. Structure content by sections
        3. Generate detailed content
        
        Args:
            topic: Topic for study guide
            
        Returns:
            Dict with study guide content
        """
        logger.info(f"Generating study guide: {topic}")
        
        # Step 1: Gather information
        info_query = f"Provide comprehensive educational content about {topic}"
        info_result = self.rag.answer_question(info_query)
        
        # Step 2: Generate structured guide
        prompt = f"""Create a comprehensive study guide about {topic} for environmental sustainability education.

Context:
{info_result['answer']}

Structure:

## Overview
Brief introduction to {topic} and its environmental significance

## Key Concepts
- List 5-7 essential concepts
- Define each clearly

## Important Facts and Statistics
- Key data points
- Evidence and research findings

## Environmental Impact
- How {topic} affects sustainability
- Current challenges
- Future implications

## Real-World Examples
- 2-3 concrete, current examples
- Case studies

## Solutions and Actions
Individual level:
- Practical actions people can take

Community/Policy level:
- Broader initiatives and policies

## Discussion Questions
- 3-5 thought-provoking questions for deeper understanding

## Further Learning
- Related topics to explore
- Key resources

Use clear headers (##) and bullet points (-) for organization."""
        
        guide_content = self.llm.generate(prompt, max_tokens=3000, temperature=0.7)
        
        return {
            "topic": topic,
            "content": guide_content,
            "metadata": {
                "used_web_search": info_result['metadata'].get('used_web_search', False),
                "num_sources": info_result['metadata'].get('num_local_sources', 0),
                "rag_confidence": info_result['metadata'].get('retrieval_confidence', 0)
            }
        }
    
    # ==================== TASK 2: AWARENESS MATERIALS ====================
    
    def generate_awareness_material(self, topic: str, format_type: str = "article",
                                   audience: str = "general public") -> Dict:
        """
        TASK 2: Generate awareness materials.
        
        Multi-step process with audience adaptation.
        
        Args:
            topic: Topic for awareness content
            format_type: article/social_media/infographic_text/poster_text
            audience: Target audience
            
        Returns:
            Dict with awareness content
        """
        logger.info(f"Generating {format_type} for {audience}: {topic}")
        
        # Step 1: Gather information
        info_query = f"Provide engaging information about {topic} for {audience} awareness"
        info_result = self.rag.answer_question(info_query)
        
        # Step 2: Multi-step generation with audience analysis
        if format_type == "article":
            return self._generate_awareness_article(topic, audience, info_result['answer'])
        elif format_type == "social_media":
            return self._generate_social_media(topic, audience, info_result['answer'])
        elif format_type == "infographic_text":
            return self._generate_infographic_text(topic, audience, info_result['answer'])
        elif format_type == "poster_text":
            return self._generate_poster_text(topic, audience, info_result['answer'])
        else:
            return self._generate_awareness_article(topic, audience, info_result['answer'])
    
    def _generate_awareness_article(self, topic: str, audience: str, context: str) -> Dict:
        """Generate awareness article with multi-step reasoning."""
        
        chain_steps = [
            {
                "name": "analyze_audience",
                "prompt": f"""Analyze the target audience for awareness content.

Topic: {topic}
Audience: {audience}

Provide analysis in JSON:
{{
  "audience_characteristics": ["char1", "char2"],
  "key_interests": ["interest1", "interest2"],
  "appropriate_tone": "tone description",
  "knowledge_level": "beginner/intermediate/advanced"
}}""",
                "temperature": 0.6,
                "output_key": "audience_analysis"
            },
            {
                "name": "identify_key_messages",
                "prompt": f"""Based on audience analysis, identify key messages.

Audience Analysis: {{audience_analysis}}
Topic: {topic}
Context: {context[:800]}

Identify 3 key messages in JSON:
{{
  "key_messages": [
    {{
      "message": "main point",
      "supporting_fact": "fact or statistic",
      "relevance": "why it matters to audience"
    }}
  ],
  "call_to_action": "suggested action"
}}""",
                "temperature": 0.6,
                "output_key": "key_messages"
            },
            {
                "name": "write_article",
                "prompt": f"""Write an engaging 500-700 word article about {topic}.

Audience Analysis: {{audience_analysis}}
Key Messages: {{key_messages}}
Context: {context[:1000]}

Article Structure:
1. Compelling introduction (hook the reader)
2. Present key messages naturally
3. Personal and global relevance
4. Include the call-to-action

Tone: Engaging, informative, actionable, hopeful
Audience: {audience}

Return only the article text:""",
                "temperature": 0.8,
                "output_key": "article"
            }
        ]
        
        try:
            result = self.llm.chain_generate(chain_steps)
            content = result.get('article', '')
            
            return {
                "topic": topic,
                "format": "article",
                "audience": audience,
                "content": content,
                "metadata": {
                    "used_chain_reasoning": True,
                    "audience_adapted": True
                }
            }
        except Exception as e:
            logger.error(f"Article chain failed: {e}")
            return self._generate_awareness_fallback(topic, audience, context, "article")
    
    def _generate_social_media(self, topic: str, audience: str, context: str) -> Dict:
        """Generate social media posts."""
        
        prompt = f"""Create 5 engaging social media posts about {topic} for {audience}.

Context: {context[:1000]}

Requirements:
- Each post under 280 characters
- Include relevant hashtags
- Mix of facts, tips, and calls-to-action
- Engaging and shareable
- Promote environmental awareness

Format:
POST 1: [text]
#hashtag1 #hashtag2

POST 2: [text]
#hashtag1 #hashtag2

...

Return all 5 posts:"""
        
        content = self.llm.generate(prompt, max_tokens=1500, temperature=0.8)
        
        return {
            "topic": topic,
            "format": "social_media",
            "audience": audience,
            "content": content,
            "metadata": {
                "used_chain_reasoning": False
            }
        }
    
    def _generate_infographic_text(self, topic: str, audience: str, context: str) -> Dict:
        """Generate infographic text."""
        
        prompt = f"""Create text content for an infographic about {topic} for {audience}.

Context: {context[:1000]}

Include:
- Catchy title (under 10 words)
- 6-8 key facts/statistics (each 1-2 lines)
- Visual data points
- One main call-to-action

Format clearly for visual design:"""
        
        content = self.llm.generate(prompt, max_tokens=1500, temperature=0.7)
        
        return {
            "topic": topic,
            "format": "infographic_text",
            "audience": audience,
            "content": content,
            "metadata": {
                "used_chain_reasoning": False
            }
        }
    
    def _generate_poster_text(self, topic: str, audience: str, context: str) -> Dict:
        """Generate poster text."""
        
        prompt = f"""Create text for an awareness poster about {topic} for {audience}.

Context: {context[:1000]}

Include:
- Bold headline (5-8 words)
- 3-4 key impact points (each 1 line)
- Strong call-to-action (1 line)
- Tagline (5-7 words)

Style: Bold, direct, memorable

Return poster text:"""
        
        content = self.llm.generate(prompt, max_tokens=1000, temperature=0.8)
        
        return {
            "topic": topic,
            "format": "poster_text",
            "audience": audience,
            "content": content,
            "metadata": {
                "used_chain_reasoning": False
            }
        }
    
    def _generate_awareness_fallback(self, topic: str, audience: str, 
                                    context: str, format_type: str) -> Dict:
        """Fallback awareness generation."""
        
        prompt = f"""Create {format_type} content about {topic} for {audience}.

Context: {context[:1500]}

Make it engaging and actionable.

Content:"""
        
        content = self.llm.generate(prompt, max_tokens=2000, temperature=0.8)
        
        return {
            "topic": topic,
            "format": format_type,
            "audience": audience,
            "content": content,
            "metadata": {
                "used_chain_reasoning": False,
                "fallback": True
            }
        }
    
    # ==================== BONUS: SELF-LEARNING ====================
    
    def provide_feedback(self, question: str, answer: str,
                        feedback_type: str, feedback_text: str,
                        rating: Optional[int] = None):
        """
        BONUS: Provide feedback for self-learning.
        
        Args:
            question: Original question
            answer: Generated answer
            feedback_type: positive/negative/suggestion
            feedback_text: Detailed feedback
            rating: Optional 1-5 rating
        """
        self.rag.record_feedback(question, answer, feedback_type, feedback_text, rating)
        logger.info(f"✓ Feedback recorded for self-learning")
    
    def get_learning_insights(self) -> Dict:
        """
        BONUS: Get insights from self-learning.
        
        Returns:
            Dict with learning insights
        """
        analysis_file = self.rag.feedback_storage / "feedback_analysis.json"
        
        if not analysis_file.exists():
            return {
                "total_feedback": 0,
                "insights": "No feedback data yet"
            }
        
        try:
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
            
            return {
                "total_feedback": analysis.get('total', 0),
                "positive_feedback": analysis.get('positive', 0),
                "negative_feedback": analysis.get('negative', 0),
                "average_rating": analysis.get('avg_rating', 0),
                "common_improvements": analysis.get('common_improvements', [])[:3],
                "insights": f"System has learned from {analysis.get('total', 0)} feedback items"
            }
        except Exception as e:
            logger.error(f"Error loading insights: {e}")
            return {
                "error": str(e)
            }
    
    # ==================== EMAIL FUNCTIONALITY ====================
    
    def send_content_via_email(self, recipient: str, content_filepath: str,
                              content_type: str, topic: str,
                              sender_email: Optional[str] = None,
                              sender_password: Optional[str] = None) -> Dict:
        """
        Send generated content via email using LangChain multi-step reasoning.
        
        Args:
            recipient: Recipient email
            content_filepath: Path to generated document
            content_type: Type of content (quiz/study_guide/report)
            topic: Content topic
            sender_email: Sender email (optional, uses env var if not provided)
            sender_password: Sender password (optional, uses env var if not provided)
            
        Returns:
            Dict with send status
        """
        from partd.email_tool import EmailTool
        
        email_tool = EmailTool(
            sender_email=sender_email,
            sender_password=sender_password,
            llm_client=self.llm
        )
        
        if not email_tool.enabled:
            return {
                "sent": False,
                "recipient": recipient,
                "content_type": content_type,
                "filepath": content_filepath,
                "error": "Email tool not enabled. Please set SENDER_EMAIL and SENDER_PASSWORD environment variables.",
                "used_langchain_reasoning": False
            }
        
        try:
            if content_type == "quiz":
                success = email_tool.send_quiz_result(recipient, content_filepath, topic)
            elif content_type == "study_guide":
                success = email_tool.send_study_guide(recipient, content_filepath, topic)
            else:
                # Generic send with LangChain-generated body
                subject = f"{content_type.replace('_', ' ').title()}: {topic.title()}"
                body = email_tool.generate_outreach_email_with_reasoning(
                    topic=f"{content_type} on {topic}",
                    audience="general public"
                )
                success = email_tool.send_document(recipient, subject, body, [content_filepath])
            
            return {
                "sent": success,
                "recipient": recipient,
                "content_type": content_type,
                "filepath": content_filepath,
                "used_langchain_reasoning": True
            }
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Email sending exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "sent": False,
                "recipient": recipient,
                "content_type": content_type,
                "filepath": content_filepath,
                "error": str(e),
                "used_langchain_reasoning": False
            }
    
    def send_outreach_email(self, to_emails: List[str], topic: str,
                           audience: str = "general public",
                           attachments: Optional[List[str]] = None,
                           sender_email: Optional[str] = None,
                           sender_password: Optional[str] = None) -> Dict:
        """
        Generate and send outreach email using LangChain multi-step reasoning.
        
        Args:
            to_emails: List of recipient emails
            topic: Email topic
            audience: Target audience
            attachments: Optional file paths to attach
            sender_email: Sender email (optional)
            sender_password: Sender password (optional)
            
        Returns:
            Dict with send results
        """
        from partd.email_tool import EmailTool
        
        email_tool = EmailTool(
            sender_email=sender_email,
            sender_password=sender_password,
            llm_client=self.llm
        )
        
        return email_tool.send_outreach_email(to_emails, topic, audience, attachments)
