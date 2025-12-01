"""
Input/Output Guardrails
Lightweight security measures for prompt injection prevention and output moderation.
"""

import re
import logging
from typing import Dict, Tuple, Optional
from html import escape

logger = logging.getLogger(__name__)


class InputGuardrails:
    """Input sanitization and prompt injection detection."""
    
    # Common prompt injection patterns
    INJECTION_PATTERNS = [
        r'ignore\s+(previous|above|all)\s+(instructions?|prompts?|commands?)',
        r'forget\s+(previous|above|all)',
        r'you\s+are\s+now\s+(a|an)\s+',
        r'act\s+as\s+(if\s+)?(you\s+are\s+)?',
        r'system\s*:\s*',
        r'<\|(system|user|assistant)\|>',
        r'\[INST\]|\[/INST\]',
        r'###\s*(system|instruction|command)',
        r'override|bypass|hack',
        r'disregard\s+(previous|above|all)',
    ]
    
    # Suspicious command patterns
    COMMAND_PATTERNS = [
        r'exec\(|eval\(|__import__|compile\(',
        r'subprocess|os\.system|shell\s*=\s*true',
        r'rm\s+-rf|del\s+/f|format\s+c:',
    ]
    
    MAX_INPUT_LENGTH = 5000  # characters
    
    @classmethod
    def sanitize_input(cls, user_input: str) -> Tuple[str, Dict[str, any]]:
        """
        Sanitize and validate user input.
        
        Args:
            user_input: Raw user input
            
        Returns:
            Tuple of (sanitized_input, validation_result)
        """
        if not user_input or not isinstance(user_input, str):
            return "", {"valid": False, "reason": "Empty or invalid input"}
        
        original_length = len(user_input)
        sanitized = user_input.strip()
        
        validation_result = {
            "valid": True,
            "warnings": [],
            "sanitized": True,
            "original_length": original_length
        }
        
        # Check length
        if len(sanitized) > cls.MAX_INPUT_LENGTH:
            sanitized = sanitized[:cls.MAX_INPUT_LENGTH]
            validation_result["warnings"].append(f"Input truncated from {original_length} to {cls.MAX_INPUT_LENGTH} characters")
            validation_result["sanitized"] = True
            logger.warning(f"Input truncated: {original_length} -> {cls.MAX_INPUT_LENGTH} chars")
        
        # Check for prompt injection patterns
        injection_detected = False
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, sanitized, re.IGNORECASE):
                injection_detected = True
                validation_result["warnings"].append(f"Potential prompt injection detected: {pattern}")
                logger.warning(f"Potential prompt injection detected in input: {pattern[:50]}")
                break
        
        # Check for command injection patterns
        command_detected = False
        for pattern in cls.COMMAND_PATTERNS:
            if re.search(pattern, sanitized, re.IGNORECASE):
                command_detected = True
                validation_result["warnings"].append(f"Potential command injection detected: {pattern}")
                logger.warning(f"Potential command injection detected in input: {pattern[:50]}")
                break
        
        # If critical threats detected, mark as suspicious but don't block (log for monitoring)
        if command_detected:
            validation_result["suspicious"] = True
            validation_result["severity"] = "high"
        elif injection_detected:
            validation_result["suspicious"] = True
            validation_result["severity"] = "medium"
        
        # Basic HTML/script tag removal (if present)
        if '<script' in sanitized.lower() or '<iframe' in sanitized.lower():
            sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
            sanitized = re.sub(r'<iframe[^>]*>.*?</iframe>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
            validation_result["warnings"].append("HTML/script tags removed")
            validation_result["sanitized"] = True
        
        validation_result["final_length"] = len(sanitized)
        
        return sanitized, validation_result


class OutputGuardrails:
    """Output moderation and content filtering."""
    
    # Basic profanity patterns (common words, can be expanded)
    PROFANITY_PATTERNS = [
        r'\b(fuck|shit|damn|hell|bitch|asshole)\w*\b',
    ]
    
    # Potentially harmful content indicators
    HARMFUL_PATTERNS = [
        r'how\s+to\s+(make|create|build)\s+(a\s+)?(bomb|weapon|drug)',
        r'kill\s+(yourself|myself|himself|herself)',
        r'commit\s+suicide',
    ]
    
    MAX_OUTPUT_LENGTH = 10000  # characters
    
    @classmethod
    def moderate_output(cls, output: str, content_type: str = "text") -> Tuple[str, Dict[str, any]]:
        """
        Moderate and validate output content.
        
        Args:
            output: Generated output content
            content_type: Type of content (text, json, etc.)
            
        Returns:
            Tuple of (moderated_output, moderation_result)
        """
        if not output or not isinstance(output, str):
            return "", {"valid": False, "reason": "Empty or invalid output"}
        
        original_length = len(output)
        moderated = output
        
        moderation_result = {
            "valid": True,
            "warnings": [],
            "moderated": False,
            "original_length": original_length
        }
        
        # Check length
        if len(moderated) > cls.MAX_OUTPUT_LENGTH:
            moderated = moderated[:cls.MAX_OUTPUT_LENGTH]
            moderation_result["warnings"].append(f"Output truncated from {original_length} to {cls.MAX_OUTPUT_LENGTH} characters")
            moderation_result["moderated"] = True
            logger.warning(f"Output truncated: {original_length} -> {cls.MAX_OUTPUT_LENGTH} chars")
        
        # Check for profanity (log but don't block - educational content may need context)
        profanity_detected = False
        for pattern in cls.PROFANITY_PATTERNS:
            if re.search(pattern, moderated, re.IGNORECASE):
                profanity_detected = True
                moderation_result["warnings"].append(f"Profanity detected: {pattern}")
                logger.warning(f"Profanity detected in output: {pattern[:50]}")
                break
        
        # Check for harmful content
        harmful_detected = False
        for pattern in cls.HARMFUL_PATTERNS:
            if re.search(pattern, moderated, re.IGNORECASE):
                harmful_detected = True
                moderation_result["warnings"].append(f"Potentially harmful content detected: {pattern}")
                moderation_result["severity"] = "high"
                logger.error(f"Potentially harmful content detected in output: {pattern[:50]}")
                # For harmful content, replace with safe message
                moderated = "I cannot provide information on this topic. Please ask about environmental sustainability instead."
                moderation_result["moderated"] = True
                moderation_result["blocked"] = True
                break
        
        if profanity_detected:
            moderation_result["severity"] = "low"
        
        moderation_result["final_length"] = len(moderated)
        
        return moderated, moderation_result
    
    @classmethod
    def escape_html(cls, text: str) -> str:
        """Escape HTML characters for safe display."""
        if not text:
            return ""
        return escape(str(text))


def apply_input_guardrails(user_input: str) -> Tuple[str, Dict]:
    """
    Apply input guardrails to user input.
    
    Args:
        user_input: Raw user input
        
    Returns:
        Tuple of (sanitized_input, validation_result)
    """
    try:
        return InputGuardrails.sanitize_input(user_input)
    except Exception as e:
        logger.error(f"Error in input guardrails: {e}")
        # On error, return original input with warning
        return user_input, {"valid": True, "warnings": [f"Guardrail error: {str(e)}"], "error": True}


def apply_output_guardrails(output: str, content_type: str = "text") -> Tuple[str, Dict]:
    """
    Apply output guardrails to generated content.
    
    Args:
        output: Generated output
        content_type: Type of content
        
    Returns:
        Tuple of (moderated_output, moderation_result)
    """
    try:
        return OutputGuardrails.moderate_output(output, content_type)
    except Exception as e:
        logger.error(f"Error in output guardrails: {e}")
        # On error, return original output with warning
        return output, {"valid": True, "warnings": [f"Guardrail error: {str(e)}"], "error": True}

