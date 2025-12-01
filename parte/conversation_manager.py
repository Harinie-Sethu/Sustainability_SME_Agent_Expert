"""
Conversation Manager for Context and Memory Management
Handles conversation history, context windows, and user preferences
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, List, Optional, Any
import json
import logging
from datetime import datetime
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation context and memory with:
    - Conversation history tracking
    - Context window management
    - User preference memory
    - Session management
    """
    
    def __init__(self, max_history: int = 20, 
                 context_window: int = 5,
                 memory_file: Optional[str] = None):
        """
        Initialize conversation manager.
        
        Args:
            max_history: Maximum messages to keep in history
            context_window: Number of recent messages for context
            memory_file: Path to persistent memory storage
        """
        self.max_history = max_history
        self.context_window = context_window
        self.memory_file = Path(memory_file) if memory_file else Path("parte/conversation_memory.json")
        
        # Conversation state
        self.conversation_history: deque = deque(maxlen=max_history)
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.user_preferences: Dict[str, Any] = {}
        self.context_cache: Dict[str, Any] = {}
        
        # Load persistent memory if exists
        self._load_memory()
        
        logger.info(f" Conversation manager initialized (session: {self.current_session_id})")
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        Add message to conversation history.
        
        Args:
            role: 'user' or 'agent'
            content: Message content
            metadata: Optional metadata (task_type, handler_used, etc.)
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.conversation_history.append(message)
        logger.info(f"Added {role} message to history (total: {len(self.conversation_history)})")
    
    def get_recent_context(self, num_messages: Optional[int] = None) -> List[Dict]:
        """
        Get recent conversation for context.
        
        Args:
            num_messages: Number of recent messages (uses context_window if None)
            
        Returns:
            List of recent messages
        """
        n = num_messages or self.context_window
        recent = list(self.conversation_history)[-n:]
        return recent
    
    def get_formatted_history(self, num_messages: Optional[int] = None) -> str:
        """
        Get formatted conversation history as string.
        
        Args:
            num_messages: Number of recent messages
            
        Returns:
            Formatted history string
        """
        recent = self.get_recent_context(num_messages)
        
        if not recent:
            return "No previous conversation."
        
        formatted = []
        for msg in recent:
            role = msg['role'].title()
            content = msg['content']
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
    
    def get_context_summary(self, llm_client=None) -> str:
        """
        Get intelligent summary of conversation context.
        
        Args:
            llm_client: Optional LLM client for summarization
            
        Returns:
            Context summary
        """
        recent = self.get_recent_context(self.max_history)
        
        if not recent:
            return "New conversation with no prior context."
        
        # If LLM available, use it for smart summarization
        if llm_client:
            from parte.prompts import CONTEXT_SUMMARY_PROMPT
            
            full_history = self.get_formatted_history(self.max_history)
            prompt = CONTEXT_SUMMARY_PROMPT.format(full_history=full_history)
            
            try:
                summary = llm_client.generate(prompt, max_tokens=300, temperature=0.5)
                return summary.strip()
            except Exception as e:
                logger.warning(f"LLM summarization failed: {e}")
        
        # Fallback: Simple summarization
        topics = set()
        for msg in recent:
            if msg['role'] == 'user':
                # Extract key topics (simple approach)
                words = msg['content'].lower().split()
                keywords = [w for w in words if len(w) > 5][:3]
                topics.update(keywords)
        
        return f"Conversation topics: {', '.join(list(topics)[:5])}"
    
    def update_user_preference(self, key: str, value: Any):
        """
        Store user preference.
        
        Args:
            key: Preference key
            value: Preference value
        """
        self.user_preferences[key] = value
        logger.info(f"Updated user preference: {key} = {value}")
        self._save_memory()
    
    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference with default."""
        return self.user_preferences.get(key, default)
    
    def cache_context(self, key: str, value: Any):
        """
        Cache context data for the session.
        
        Args:
            key: Cache key
            value: Data to cache
        """
        self.context_cache[key] = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_cached_context(self, key: str) -> Optional[Any]:
        """Retrieve cached context."""
        cached = self.context_cache.get(key)
        return cached['value'] if cached else None
    
    def clear_session(self):
        """Clear current session data."""
        self.conversation_history.clear()
        self.context_cache.clear()
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Session cleared. New session: {self.current_session_id}")
    
    def export_conversation(self, filepath: Optional[str] = None) -> str:
        """
        Export conversation history to file.
        
        Args:
            filepath: Optional file path
            
        Returns:
            Path to exported file
        """
        if not filepath:
            filepath = f"parte/conversation_export_{self.current_session_id}.json"
        
        export_data = {
            "session_id": self.current_session_id,
            "exported_at": datetime.now().isoformat(),
            "conversation": list(self.conversation_history),
            "user_preferences": self.user_preferences
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f" Conversation exported to {filepath}")
        return filepath
    
    def _save_memory(self):
        """Save persistent memory to file."""
        memory_data = {
            "user_preferences": self.user_preferences,
            "last_session": self.current_session_id,
            "updated_at": datetime.now().isoformat()
        }
        
        self.memory_file.parent.mkdir(exist_ok=True, parents=True)
        
        with open(self.memory_file, 'w') as f:
            json.dump(memory_data, f, indent=2)
    
    def _load_memory(self):
        """Load persistent memory from file."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    memory_data = json.load(f)
                
                self.user_preferences = memory_data.get('user_preferences', {})
                logger.info(f" Loaded user preferences: {len(self.user_preferences)} items")
            except Exception as e:
                logger.warning(f"Could not load memory: {e}")
    
    def get_conversation_stats(self) -> Dict:
        """Get statistics about current conversation."""
        return {
            "session_id": self.current_session_id,
            "total_messages": len(self.conversation_history),
            "user_messages": sum(1 for m in self.conversation_history if m['role'] == 'user'),
            "agent_messages": sum(1 for m in self.conversation_history if m['role'] == 'agent'),
            "context_items_cached": len(self.context_cache),
            "user_preferences": len(self.user_preferences)
        }
