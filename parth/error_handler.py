"""
Error Handler with Multi-Call Recovery
Implements retry logic, fallbacks, and error recovery strategies
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, Any, Callable, Optional, List
import logging
import time
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorHandler:
    """
    Multi-call recovery and error handling:
    - Retry with exponential backoff
    - Fallback strategies
    - Error categorization
    - Recovery suggestions
    """
    
    def __init__(self):
        """Initialize error handler."""
        self.error_log: List[Dict] = []
        logger.info(" Error Handler initialized")
    
    def retry_with_backoff(self,
                          func: Callable,
                          max_retries: int = 3,
                          backoff_factor: float = 2.0,
                          exceptions: tuple = (Exception,),
                          **kwargs) -> Dict[str, Any]:
        """
        Execute function with retry and exponential backoff.
        
        Args:
            func: Function to execute
            max_retries: Maximum retry attempts
            backoff_factor: Backoff multiplier
            exceptions: Exceptions to catch
            **kwargs: Function arguments
            
        Returns:
            Execution result
        """
        attempt = 0
        last_error = None
        
        while attempt < max_retries:
            try:
                result = func(**kwargs)
                
                if attempt > 0:
                    logger.info(f" Succeeded after {attempt} retries")
                
                return {
                    'success': True,
                    'result': result,
                    'attempts': attempt + 1
                }
            
            except exceptions as e:
                attempt += 1
                last_error = e
                
                if attempt < max_retries:
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} attempts failed")
        
        # Log error
        self._log_error(func.__name__, last_error, max_retries)
        
        return {
            'success': False,
            'error': str(last_error),
            'attempts': max_retries
        }
    
    def execute_with_fallback(self,
                             primary_func: Callable,
                             fallback_func: Callable,
                             primary_kwargs: Dict[str, Any],
                             fallback_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute function with fallback.
        
        Args:
            primary_func: Primary function to try
            fallback_func: Fallback function if primary fails
            primary_kwargs: Arguments for primary function
            fallback_kwargs: Arguments for fallback (defaults to primary_kwargs)
            
        Returns:
            Execution result
        """
        logger.info(f"Executing {primary_func.__name__} with fallback to {fallback_func.__name__}")
        
        # Try primary
        try:
            result = primary_func(**primary_kwargs)
            return {
                'success': True,
                'result': result,
                'used_fallback': False,
                'function': primary_func.__name__
            }
        
        except Exception as e:
            logger.warning(f"Primary function failed: {e}")
            logger.info(f"Attempting fallback: {fallback_func.__name__}")
            
            # Try fallback
            try:
                fallback_kwargs = fallback_kwargs or primary_kwargs
                result = fallback_func(**fallback_kwargs)
                
                return {
                    'success': True,
                    'result': result,
                    'used_fallback': True,
                    'function': fallback_func.__name__,
                    'primary_error': str(e)
                }
            
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                
                self._log_error(primary_func.__name__, e, 1)
                self._log_error(fallback_func.__name__, fallback_error, 1)
                
                return {
                    'success': False,
                    'primary_error': str(e),
                    'fallback_error': str(fallback_error)
                }
    
    def execute_with_multiple_fallbacks(self,
                                       functions: List[tuple],
                                       stop_on_success: bool = True) -> Dict[str, Any]:
        """
        Execute functions in order until one succeeds.
        
        Args:
            functions: List of (function, kwargs) tuples
            stop_on_success: Stop after first success
            
        Returns:
            Execution result
        """
        results = []
        errors = []
        
        for i, (func, kwargs) in enumerate(functions):
            logger.info(f"Attempting function {i+1}/{len(functions)}: {func.__name__}")
            
            try:
                result = func(**kwargs)
                results.append({
                    'function': func.__name__,
                    'success': True,
                    'result': result
                })
                
                if stop_on_success:
                    logger.info(f" Success with {func.__name__}")
                    return {
                        'success': True,
                        'result': result,
                        'function': func.__name__,
                        'attempt_number': i + 1,
                        'all_results': results
                    }
            
            except Exception as e:
                logger.warning(f"Function {func.__name__} failed: {e}")
                errors.append({
                    'function': func.__name__,
                    'error': str(e)
                })
                results.append({
                    'function': func.__name__,
                    'success': False,
                    'error': str(e)
                })
        
        # All failed
        logger.error(f"All {len(functions)} functions failed")
        
        return {
            'success': False,
            'errors': errors,
            'all_results': results
        }
    
    def _log_error(self, function_name: str, error: Exception, attempts: int):
        """Log error for analysis."""
        self.error_log.append({
            'timestamp': time.time(),
            'function': function_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'attempts': attempts
        })
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        if not self.error_log:
            return {'message': 'No errors logged'}
        
        error_types = {}
        function_errors = {}
        
        for log_entry in self.error_log:
            error_type = log_entry['error_type']
            function = log_entry['function']
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
            function_errors[function] = function_errors.get(function, 0) + 1
        
        return {
            'total_errors': len(self.error_log),
            'error_types': error_types,
            'function_errors': function_errors,
            'most_common_error': max(error_types.items(), key=lambda x: x[1]) if error_types else None
        }
    
    def clear_error_log(self):
        """Clear error log."""
        self.error_log = []
        logger.info("Error log cleared")


def with_retry(max_retries: int = 3, backoff: float = 2.0):
    """
    Decorator for automatic retry with backoff.
    
    Args:
        max_retries: Maximum retry attempts
        backoff: Backoff multiplier
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = ErrorHandler()
            return handler.retry_with_backoff(
                func,
                max_retries=max_retries,
                backoff_factor=backoff,
                **kwargs
            )
        return wrapper
    return decorator


def with_fallback(fallback_func: Callable):
    """
    Decorator for automatic fallback.
    
    Args:
        fallback_func: Fallback function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = ErrorHandler()
            return handler.execute_with_fallback(
                func,
                fallback_func,
                kwargs,
                kwargs
            )
        return wrapper
    return decorator
