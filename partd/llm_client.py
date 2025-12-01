"""
LLM Client with LangChain integration for multi-step reasoning
Combines Gemini API with RAG capabilities
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import google.generativeai as genai
import os
from typing import Dict, Optional, List, Any
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiLLMClient:
    """
    Enhanced Gemini client with multi-step reasoning support.
    Integrates direct API calls with LangChain-style chaining.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key (or use GEMINI_API_KEY env var)
            model: Model name
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. "
                "Set GEMINI_API_KEY environment variable. "
                "Get API key from: https://makersuite.google.com/app/apikey"
            )
        
        genai.configure(api_key=self.api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(self.model_name)
        
        logger.info(f"✓ Gemini LLM client initialized: {self.model_name}")
    
    def generate(self, prompt: str, max_tokens: int = 2048, 
                temperature: float = 0.7) -> str:
        """
        Generate text response.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum response length
            temperature: Creativity level (0=focused, 1=creative)
            
        Returns:
            Generated text
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                )
            )
            return response.text
        
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_json(self, prompt: str, max_tokens: int = 2048) -> Dict:
        """
        Generate JSON response with cleanup.
        
        Args:
            prompt: Input prompt requesting JSON output
            max_tokens: Maximum response length
            
        Returns:
            Parsed JSON dictionary
        """
        # Add explicit JSON formatting instruction
        json_prompt = f"""{prompt}

CRITICAL: Return ONLY valid JSON. No markdown code blocks, no explanations.
Start with {{ or [."""
        
        response_text = self.generate(json_prompt, max_tokens=max_tokens, temperature=0.3)
        
        # Clean response
        response_text = response_text.strip()
        
        # Remove markdown code blocks
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response was: {response_text[:500]}")
            return {"error": "Failed to parse JSON", "raw_response": response_text}
    
    def generate_with_context(self, system_prompt: str, user_message: str,
                             max_tokens: int = 2048, temperature: float = 0.7) -> str:
        """
        Generate response with system context.
        
        Args:
            system_prompt: System instructions/context
            user_message: User's message
            max_tokens: Maximum response length
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        combined_prompt = f"""{system_prompt}

User Query: {user_message}

Response:"""
        
        return self.generate(combined_prompt, max_tokens=max_tokens, temperature=temperature)
    
    def chain_generate(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Multi-step chain generation (LangChain-style).
        
        Args:
            steps: List of step configurations:
                [
                    {
                        "name": "step_name",
                        "prompt": "prompt with {variable} placeholders",
                        "temperature": 0.7,
                        "output_key": "key_to_store_result"
                    }
                ]
        
        Returns:
            Dict with all step outputs
        """
        results = {}
        
        for step in steps:
            step_name = step.get("name", "unnamed_step")
            prompt_template = step.get("prompt", "")
            output_key = step.get("output_key", step_name)
            temperature = step.get("temperature", 0.7)
            
            # Replace variables in prompt with previous results
            prompt = prompt_template
            for key, value in results.items():
                placeholder = "{" + key + "}"
                if placeholder in prompt:
                    prompt = prompt.replace(placeholder, str(value))
            
            # Generate
            logger.info(f"Executing chain step: {step_name}")
            output = self.generate(prompt, temperature=temperature)
            
            results[output_key] = output
        
        return results
