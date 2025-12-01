"""
Tool Registry
Central registry for all available tools
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, Any, Callable, Optional, List
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Central registry for all tools with:
    - Tool registration
    - Capability tracking
    - Usage statistics
    - Tool metadata
    """
    
    def __init__(self):
        """Initialize tool registry."""
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.usage_stats: Dict[str, int] = {}
        self.tool_categories: Dict[str, List[str]] = {
            'knowledge': [],
            'generation': [],
            'export': [],
            'communication': [],
            'analysis': []
        }
        
        logger.info(" Tool Registry initialized")
    
    def register_tool(self, 
                     tool_name: str,
                     tool_function: Callable,
                     category: str,
                     description: str,
                     required_params: List[str],
                     optional_params: Optional[List[str]] = None,
                     fallback_tool: Optional[str] = None,
                     retry_config: Optional[Dict] = None):
        """
        Register a tool.
        
        Args:
            tool_name: Unique tool identifier
            tool_function: Function to execute
            category: Tool category
            description: Tool description
            required_params: Required parameters
            optional_params: Optional parameters
            fallback_tool: Fallback tool if this fails
            retry_config: Retry configuration
        """
        self.tools[tool_name] = {
            'function': tool_function,
            'category': category,
            'description': description,
            'required_params': required_params,
            'optional_params': optional_params or [],
            'fallback_tool': fallback_tool,
            'retry_config': retry_config or {'max_retries': 3, 'backoff': 2},
            'registered_at': datetime.now().isoformat()
        }
        
        # Add to category
        if category in self.tool_categories:
            self.tool_categories[category].append(tool_name)
        
        # Initialize usage stats
        self.usage_stats[tool_name] = 0
        
        logger.info(f" Registered tool: {tool_name} ({category})")
    
    def get_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool by name."""
        return self.tools.get(tool_name)
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool.
        
        Args:
            tool_name: Tool to execute
            **kwargs: Tool parameters
            
        Returns:
            Execution result
        """
        if tool_name not in self.tools:
            return {
                'success': False,
                'error': f"Tool '{tool_name}' not found"
            }
        
        tool = self.tools[tool_name]
        
        # Validate required parameters
        missing_params = [p for p in tool['required_params'] if p not in kwargs]
        if missing_params:
            return {
                'success': False,
                'error': f"Missing required parameters: {missing_params}"
            }
        
        # Execute tool
        try:
            result = tool['function'](**kwargs)
            self.usage_stats[tool_name] += 1
            
            return {
                'success': True,
                'result': result,
                'tool': tool_name
            }
        
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            return {
                'success': False,
                'error': str(e),
                'tool': tool_name
            }
    
    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """List available tools."""
        if category:
            return self.tool_categories.get(category, [])
        return list(self.tools.keys())
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool information."""
        tool = self.tools.get(tool_name)
        if not tool:
            return None
        
        return {
            'name': tool_name,
            'category': tool['category'],
            'description': tool['description'],
            'required_params': tool['required_params'],
            'optional_params': tool['optional_params'],
            'fallback_tool': tool['fallback_tool'],
            'usage_count': self.usage_stats.get(tool_name, 0)
        }
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get tool usage statistics."""
        return {
            'total_tools': len(self.tools),
            'tools_by_category': {cat: len(tools) for cat, tools in self.tool_categories.items()},
            'usage_stats': self.usage_stats,
            'most_used': sorted(self.usage_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def save_registry(self, filepath: str):
        """Save registry metadata (not functions)."""
        registry_data = {
            'tools': {
                name: {
                    'category': tool['category'],
                    'description': tool['description'],
                    'required_params': tool['required_params'],
                    'optional_params': tool['optional_params'],
                    'fallback_tool': tool['fallback_tool'],
                    'registered_at': tool['registered_at']
                }
                for name, tool in self.tools.items()
            },
            'usage_stats': self.usage_stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        logger.info(f" Registry saved to {filepath}")
