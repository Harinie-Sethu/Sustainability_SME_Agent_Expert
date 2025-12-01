"""
Comprehensive Prompt Library for Environmental SME
Organized prompts for different tasks with variations
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class PromptLibrary:
    """
    Organized library of prompts for different tasks.
    Supports versioning, tracking, and systematic evaluation.
    """
    
    def __init__(self, storage_dir: str = "partf/experiments/prompts"):
        """
        Initialize prompt library.
        
        Args:
            storage_dir: Directory to store prompt templates
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        
        # Prompt templates organized by task
        self.prompts = {
            "qa": self._init_qa_prompts(),
            "quiz_generation": self._init_quiz_prompts(),
            "study_guide": self._init_study_guide_prompts(),
            "awareness_content": self._init_awareness_prompts(),
            "reasoning": self._init_reasoning_prompts(),
            "summarization": self._init_summarization_prompts()
        }
        
        # Track prompt versions and performance
        self.prompt_versions: Dict[str, List[Dict]] = {}
        self.performance_log: List[Dict] = []
    
    def _init_qa_prompts(self) -> Dict[str, Dict[str, Any]]:
        """Initialize Q&A prompt templates."""
        return {
            "basic": {
                "version": "1.0",
                "template": "Answer the following question about {topic}:\n\nQuestion: {question}\n\nAnswer:",
                "description": "Basic zero-shot Q&A",
                "parameters": ["topic", "question"]
            },
            
            "contextual": {
                "version": "1.0",
                "template": """You are an expert in Environmental Sustainability.

Context: {context}

Question: {question}

Provide a clear, accurate answer based on the context above:""",
                "description": "Q&A with context",
                "parameters": ["context", "question"]
            },
            
            "chain_of_thought": {
                "version": "1.0",
                "template": """You are an expert in Environmental Sustainability.

Question: {question}

Let's think step by step:
1. First, understand what is being asked
2. Identify the key concepts involved
3. Connect relevant information
4. Provide a comprehensive answer

Answer:""",
                "description": "Q&A with chain-of-thought reasoning",
                "parameters": ["question"]
            },
            
            "structured": {
                "version": "1.0",
                "template": """Question: {question}

Provide a structured answer in the following format:

**Overview**: Brief 1-2 sentence overview

**Key Points**:
- Point 1
- Point 2
- Point 3

**Explanation**: Detailed explanation

**Actionable Insights**: What can be done about this

Answer:""",
                "description": "Structured response format",
                "parameters": ["question"]
            },
            
            "with_confidence": {
                "version": "1.0",
                "template": """Question: {question}

Answer this question and indicate your confidence level (High/Medium/Low).

Format:
Answer: [your answer]
Confidence: [High/Medium/Low]
Reasoning: [why this confidence level]

Response:""",
                "description": "Q&A with confidence indication",
                "parameters": ["question"]
            }
        }
    
    def _init_quiz_prompts(self) -> Dict[str, Dict[str, Any]]:
        """Initialize quiz generation prompt templates."""
        return {
            "basic": {
                "version": "1.0",
                "template": """Generate {num_questions} multiple-choice questions about {topic}.

Format each question as:
Question: [question text]
A) [option]
B) [option]
C) [option]
D) [option]
Correct: [letter]
Explanation: [why this is correct]

Questions:""",
                "description": "Basic quiz generation",
                "parameters": ["num_questions", "topic"]
            },
            
            "difficulty_aware": {
                "version": "1.0",
                "template": """Generate {num_questions} {difficulty}-level multiple-choice questions about {topic}.

Difficulty Guidelines:
- Easy: Basic concepts, straightforward questions
- Medium: Application of concepts, some critical thinking
- Hard: Complex scenarios, deep understanding required

Format as JSON array:
[
  {{
    "question": "question text",
    "options": {{"A": "opt1", "B": "opt2", "C": "opt3", "D": "opt4"}},
    "correct_answer": "A",
    "explanation": "explanation",
    "difficulty": "{difficulty}"
  }}
]

Generate:""",
                "description": "Difficulty-aware quiz generation",
                "parameters": ["num_questions", "topic", "difficulty"]
            },
            
            "bloom_taxonomy": {
                "version": "1.0",
                "template": """Generate {num_questions} questions about {topic} following Bloom's Taxonomy levels.

Include questions at different cognitive levels:
1. Remember (recall facts)
2. Understand (explain concepts)
3. Apply (use in new situations)
4. Analyze (examine relationships)

Context: {context}

Format each question with its Bloom's level indicated.

Questions:""",
                "description": "Bloom's taxonomy-based quiz",
                "parameters": ["num_questions", "topic", "context"]
            },
            
            "scenario_based": {
                "version": "1.0",
                "template": """Create {num_questions} scenario-based questions about {topic}.

Each question should:
1. Present a realistic scenario
2. Require application of knowledge
3. Have plausible distractors
4. Include detailed explanation

Context: {context}

Format as JSON.

Questions:""",
                "description": "Scenario-based quiz questions",
                "parameters": ["num_questions", "topic", "context"]
            }
        }
    
    def _init_study_guide_prompts(self) -> Dict[str, Dict[str, Any]]:
        """Initialize study guide prompt templates."""
        return {
            "comprehensive": {
                "version": "1.0",
                "template": """Create a comprehensive study guide on {topic}.

Include:
## Overview
## Key Concepts (5-7 concepts with definitions)
## Important Facts
## Real-World Examples
## Practice Questions
## Further Reading

Study Guide:""",
                "description": "Comprehensive study guide",
                "parameters": ["topic"]
            },
            
            "modular": {
                "version": "1.0",
                "template": """Create a modular study guide on {topic} with the following sections:

{sections}

Context: {context}

For each section, provide clear, educational content suitable for {audience}.

Study Guide:""",
                "description": "Modular, customizable study guide",
                "parameters": ["topic", "sections", "context", "audience"]
            },
            
            "visual_focused": {
                "version": "1.0",
                "template": """Create a study guide on {topic} optimized for visual learning.

Include:
- Diagrams descriptions (what should be visualized)
- Flow charts (process descriptions)
- Comparison tables
- Infographic content
- Mind map structure

Study Guide:""",
                "description": "Visual learning-focused guide",
                "parameters": ["topic"]
            }
        }
    
    def _init_awareness_prompts(self) -> Dict[str, Dict[str, Any]]:
        """Initialize awareness content prompt templates."""
        return {
            "article": {
                "version": "1.0",
                "template": """Write an engaging {word_count}-word article about {topic} for {audience}.

Tone: {tone}
Goal: Raise awareness and inspire action

Structure:
1. Hook (engaging opening)
2. Problem/Challenge
3. Impact (personal and global)
4. Solutions
5. Call to action

Article:""",
                "description": "Awareness article generation",
                "parameters": ["word_count", "topic", "audience", "tone"]
            },
            
            "social_media": {
                "version": "1.0",
                "template": """Create {num_posts} social media posts about {topic}.

Guidelines:
- Each post under 280 characters
- Include relevant hashtags
- Mix facts, questions, and calls-to-action
- Engaging and shareable
- Promote environmental awareness

Posts:""",
                "description": "Social media content",
                "parameters": ["num_posts", "topic"]
            },
            
            "storytelling": {
                "version": "1.0",
                "template": """Tell a compelling story about {topic} that illustrates its importance.

Story elements:
- Relatable protagonist
- Challenge related to {topic}
- Journey and realizations
- Positive transformation
- Inspiring conclusion

Target audience: {audience}

Story:""",
                "description": "Story-based awareness content",
                "parameters": ["topic", "audience"]
            }
        }
    
    def _init_reasoning_prompts(self) -> Dict[str, Dict[str, Any]]:
        """Initialize reasoning prompt templates."""
        return {
            "step_by_step": {
                "version": "1.0",
                "template": """Analyze this problem step by step:

Problem: {problem}

Step-by-step analysis:
1. Understanding: What is the problem asking?
2. Context: What do we know?
3. Analysis: Break down the components
4. Synthesis: Connect the pieces
5. Conclusion: What's the answer?

Analysis:""",
                "description": "Step-by-step reasoning",
                "parameters": ["problem"]
            },
            
            "comparative": {
                "version": "1.0",
                "template": """Compare {option_a} and {option_b} in the context of {context}.

Analysis framework:
1. Similarities
2. Differences
3. Pros and cons of each
4. Environmental impact
5. Recommendation

Comparison:""",
                "description": "Comparative reasoning",
                "parameters": ["option_a", "option_b", "context"]
            },
            
            "causal": {
                "version": "1.0",
                "template": """Analyze the causal relationship: {relationship}

Examine:
1. Direct causes
2. Contributing factors
3. Chain of effects
4. Feedback loops
5. Long-term implications

Analysis:""",
                "description": "Causal reasoning",
                "parameters": ["relationship"]
            }
        }
    
    def _init_summarization_prompts(self) -> Dict[str, Dict[str, Any]]:
        """Initialize summarization prompt templates."""
        return {
            "extractive": {
                "version": "1.0",
                "template": """Summarize the following text in {num_sentences} sentences:

Text: {text}

Summary:""",
                "description": "Extractive summarization",
                "parameters": ["text", "num_sentences"]
            },
            
            "abstractive": {
                "version": "1.0",
                "template": """Read and understand this text, then create an abstractive summary in your own words:

Text: {text}

Create a {length} summary that captures the essence and key insights.

Summary:""",
                "description": "Abstractive summarization",
                "parameters": ["text", "length"]
            },
            
            "bullet_points": {
                "version": "1.0",
                "template": """Summarize this text as bullet points:

Text: {text}

Key Points (5-7 bullets):
-""",
                "description": "Bullet point summary",
                "parameters": ["text"]
            }
        }
    
    def get_prompt(self, task: str, variant: str, **kwargs) -> str:
        """
        Get a prompt template filled with parameters.
        
        Args:
            task: Task type (qa, quiz_generation, etc.)
            variant: Prompt variant (basic, contextual, etc.)
            **kwargs: Parameters to fill in the template
            
        Returns:
            Formatted prompt
        """
        if task not in self.prompts:
            raise ValueError(f"Unknown task: {task}")
        
        if variant not in self.prompts[task]:
            raise ValueError(f"Unknown variant '{variant}' for task '{task}'")
        
        template_info = self.prompts[task][variant]
        template = template_info["template"]
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing parameter {e} for prompt {task}/{variant}")
    
    def get_prompt_info(self, task: str, variant: str) -> Dict[str, Any]:
        """Get prompt metadata."""
        return self.prompts[task][variant]
    
    def list_prompts(self, task: Optional[str] = None) -> Dict[str, List[str]]:
        """List available prompts."""
        if task:
            return {task: list(self.prompts[task].keys())}
        return {task: list(variants.keys()) for task, variants in self.prompts.items()}
    
    def save_prompts(self):
        """Save prompt library to disk."""
        for task, variants in self.prompts.items():
            filepath = self.storage_dir / f"{task}_prompts.json"
            with open(filepath, 'w') as f:
                json.dump(variants, f, indent=2)
    
    def log_performance(self, task: str, variant: str, metrics: Dict[str, Any]):
        """Log prompt performance."""
        log_entry = {
            "timestamp": str(Path(__file__).stat().st_mtime),
            "task": task,
            "variant": variant,
            "metrics": metrics
        }
        self.performance_log.append(log_entry)
