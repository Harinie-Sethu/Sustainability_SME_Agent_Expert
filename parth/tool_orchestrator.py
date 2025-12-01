"""
Tool Orchestrator
Orchestrates multi-tool workflows with error handling
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, Any, Optional, List, Callable
import logging
from datetime import datetime
import time

from parth.tool_registry import ToolRegistry
from parth.error_handler import ErrorHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolOrchestrator:
    """
    Orchestrates multi-tool workflows:
    - Sequential tool execution
    - Parallel tool execution
    - Conditional routing
    - Error recovery
    - Workflow logging
    """
    
    def __init__(self, tool_registry: ToolRegistry, error_handler: ErrorHandler):
        """
        Initialize tool orchestrator.
        
        Args:
            tool_registry: Tool registry
            error_handler: Error handler
        """
        self.registry = tool_registry
        self.error_handler = error_handler
        self.workflow_log: List[Dict] = []
        
        logger.info("✓ Tool Orchestrator initialized")
    
    def execute_workflow(self, workflow_definition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a defined workflow.
        
        Args:
            workflow_definition: Workflow definition with steps
            
        Returns:
            Workflow execution result
        """
        workflow_name = workflow_definition.get('name', 'unnamed_workflow')
        steps = workflow_definition.get('steps', [])
        
        logger.info(f"Executing workflow: {workflow_name} ({len(steps)} steps)")
        
        workflow_context = {}
        step_results = []
        
        start_time = time.time()
        
        for i, step in enumerate(steps, 1):
            logger.info(f"  Step {i}/{len(steps)}: {step.get('name', 'unnamed')}")
            
            step_result = self._execute_step(step, workflow_context)
            step_results.append(step_result)
            
            if not step_result['success']:
                # Check if step is required
                if step.get('required', True):
                    logger.error(f"  Required step failed, aborting workflow")
                    break
                else:
                    logger.warning(f"  Optional step failed, continuing")
            
            # Update context with step output
            if step_result['success'] and step.get('output_key'):
                workflow_context[step['output_key']] = step_result.get('result')
        
        execution_time = time.time() - start_time
        
        # Determine overall success
        required_steps = [s for s in step_results if steps[step_results.index(s)].get('required', True)]
        all_required_passed = all(s['success'] for s in required_steps)
        
        workflow_result = {
            'workflow_name': workflow_name,
            'success': all_required_passed,
            'total_steps': len(steps),
            'completed_steps': len(step_results),
            'execution_time': execution_time,
            'step_results': step_results,
            'workflow_context': workflow_context
        }
        
        # Log workflow
        self._log_workflow(workflow_name, workflow_result)
        
        return workflow_result
    
    def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single workflow step.
        
        Args:
            step: Step definition
            context: Workflow context
            
        Returns:
            Step execution result
        """
        step_name = step.get('name', 'unnamed')
        tool_name = step.get('tool')
        params = step.get('params', {})
        retry_config = step.get('retry', {})
        fallback_tool = step.get('fallback_tool')
        
        # Resolve parameters from context
        resolved_params = self._resolve_params(params, context)
        
        # Execute with retry if configured
        if retry_config:
            max_retries = retry_config.get('max_retries', 3)
            backoff = retry_config.get('backoff', 2.0)
            
            tool = self.registry.get_tool(tool_name)
            if not tool:
                return {
                    'step': step_name,
                    'success': False,
                    'error': f'Tool {tool_name} not found'
                }
            
            result = self.error_handler.retry_with_backoff(
                tool['function'],
                max_retries=max_retries,
                backoff_factor=backoff,
                **resolved_params
            )
            
            return {
                'step': step_name,
                'tool': tool_name,
                **result
            }
        
        # Execute with fallback if configured
        elif fallback_tool:
            primary_tool = self.registry.get_tool(tool_name)
            fallback = self.registry.get_tool(fallback_tool)
            
            if not primary_tool or not fallback:
                return {
                    'step': step_name,
                    'success': False,
                    'error': 'Primary or fallback tool not found'
                }
            
            result = self.error_handler.execute_with_fallback(
                primary_tool['function'],
                fallback['function'],
                resolved_params,
                resolved_params
            )
            
            return {
                'step': step_name,
                'tool': tool_name,
                **result
            }
        
        # Simple execution
        else:
            result = self.registry.execute_tool(tool_name, **resolved_params)
            return {
                'step': step_name,
                'tool': tool_name,
                **result
            }
    
    def _resolve_params(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve parameters from workflow context.
        
        Args:
            params: Parameter template
            context: Workflow context
            
        Returns:
            Resolved parameters
        """
        resolved = {}
        
        for key, value in params.items():
            if isinstance(value, str) and value.startswith('$'):
                # Context variable reference
                var_name = value[1:]
                resolved[key] = context.get(var_name, value)
            else:
                resolved[key] = value
        
        return resolved
    
    def execute_parallel(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute multiple tools in parallel (simulated).
        
        Args:
            tool_calls: List of tool call definitions
            
        Returns:
            Parallel execution results
        """
        logger.info(f"Executing {len(tool_calls)} tools in parallel")
        
        results = []
        
        # Note: In production, use threading/multiprocessing
        for call in tool_calls:
            tool_name = call.get('tool')
            params = call.get('params', {})
            
            result = self.registry.execute_tool(tool_name, **params)
            results.append({
                'tool': tool_name,
                **result
            })
        
        successful = sum(1 for r in results if r['success'])
        
        return {
            'total': len(tool_calls),
            'successful': successful,
            'failed': len(tool_calls) - successful,
            'results': results
        }
    
    def execute_conditional(self, 
                          condition: Callable[[Dict], bool],
                          if_true_tool: str,
                          if_false_tool: str,
                          params: Dict[str, Any],
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tool based on condition.
        
        Args:
            condition: Condition function
            if_true_tool: Tool to execute if true
            if_false_tool: Tool to execute if false
            params: Parameters
            context: Context for condition
            
        Returns:
            Execution result
        """
        logger.info("Executing conditional tool routing")
        
        # Evaluate condition
        try:
            should_execute_true = condition(context)
        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return {
                'success': False,
                'error': f'Condition evaluation failed: {e}'
            }
        
        # Execute appropriate tool
        tool_name = if_true_tool if should_execute_true else if_false_tool
        logger.info(f"  Condition: {should_execute_true}, executing: {tool_name}")
        
        result = self.registry.execute_tool(tool_name, **params)
        
        return {
            'condition_result': should_execute_true,
            'executed_tool': tool_name,
            **result
        }
    
    def _log_workflow(self, workflow_name: str, result: Dict[str, Any]):
        """Log workflow execution."""
        self.workflow_log.append({
            'timestamp': datetime.now().isoformat(),
            'workflow_name': workflow_name,
            'success': result['success'],
            'execution_time': result['execution_time'],
            'total_steps': result['total_steps'],
            'completed_steps': result['completed_steps']
        })
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get workflow execution statistics."""
        if not self.workflow_log:
            return {'message': 'No workflows logged'}
        
        total = len(self.workflow_log)
        successful = sum(1 for log in self.workflow_log if log['success'])
        
        avg_time = sum(log['execution_time'] for log in self.workflow_log) / total
        avg_steps = sum(log['total_steps'] for log in self.workflow_log) / total
        
        return {
            'total_workflows': total,
            'successful': successful,
            'failed': total - successful,
            'success_rate': successful / total if total > 0 else 0,
            'avg_execution_time': avg_time,
            'avg_steps': avg_steps
        }
    
    def create_workflow_template(self, name: str, description: str) -> Dict[str, Any]:
        """
        Create workflow template.
        
        Args:
            name: Workflow name
            description: Workflow description
            
        Returns:
            Workflow template
        """
        return {
            'name': name,
            'description': description,
            'steps': [],
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            }
        }
    
    def add_workflow_step(self,
                         workflow: Dict[str, Any],
                         step_name: str,
                         tool_name: str,
                         params: Dict[str, Any],
                         required: bool = True,
                         output_key: Optional[str] = None,
                         retry_config: Optional[Dict] = None,
                         fallback_tool: Optional[str] = None):
        """
        Add step to workflow.
        
        Args:
            workflow: Workflow definition
            step_name: Step name
            tool_name: Tool to execute
            params: Parameters
            required: Whether step is required
            output_key: Key to store output in context
            retry_config: Retry configuration
            fallback_tool: Fallback tool
        """
        step = {
            'name': step_name,
            'tool': tool_name,
            'params': params,
            'required': required,
            'output_key': output_key
        }
        
        if retry_config:
            step['retry'] = retry_config
        
        if fallback_tool:
            step['fallback_tool'] = fallback_tool
        
        workflow['steps'].append(step)
        
        logger.info(f"Added step to workflow '{workflow['name']}': {step_name}")
