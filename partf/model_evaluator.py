"""
Model Evaluation and Comparison
Compare different models/configurations on same tasks
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluate and compare model performance across:
    - Different models
    - Different temperatures
    - Different max_tokens
    - Different prompt strategies
    """
    
    def __init__(self, results_dir: str = "partf/experiments/results"):
        """
        Initialize model evaluator.
        
        Args:
            results_dir: Directory for evaluation results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        self.evaluations: List[Dict] = []
        
        logger.info("✓ Model Evaluator initialized")
    
    def evaluate_model_config(self, llm_client, task: str, 
                             test_cases: List[Dict],
                             config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a specific model configuration.
        
        Args:
            llm_client: LLM client
            task: Task name
            test_cases: List of test cases with input/expected output
            config: Model configuration (temperature, max_tokens, etc.)
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating model config: {config}")
        
        results = []
        start_time = time.time()
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"  Test case {i+1}/{len(test_cases)}")
            
            try:
                prompt = test_case['prompt']
                expected = test_case.get('expected_output')
                
                # Generate
                case_start = time.time()
                output = llm_client.generate(
                    prompt,
                    temperature=config.get('temperature', 0.7),
                    max_tokens=config.get('max_tokens', 1000)
                )
                case_time = time.time() - case_start
                
                # Evaluate
                result = {
                    "test_case_id": i,
                    "success": True,
                    "output": output,
                    "execution_time": case_time,
                    "output_length": len(output),
                    "token_count": len(output.split())  # Approximate
                }
                
                # Calculate metrics if expected output provided
                if expected:
                    result["accuracy_score"] = self._calculate_accuracy(output, expected)
                    result["relevance_score"] = self._calculate_relevance(output, expected)
                
                results.append(result)
            
            except Exception as e:
                logger.error(f"Test case {i} failed: {e}")
                results.append({
                    "test_case_id": i,
                    "success": False,
                    "error": str(e)
                })
        
        total_time = time.time() - start_time
        
        # Calculate aggregate metrics
        successful = [r for r in results if r.get('success')]
        
        evaluation = {
            "task": task,
            "config": config,
            "timestamp": datetime.now().isoformat(),
            "num_test_cases": len(test_cases),
            "num_successful": len(successful),
            "success_rate": len(successful) / len(test_cases),
            "total_time": total_time,
            "avg_execution_time": sum(r.get('execution_time', 0) for r in successful) / len(successful) if successful else 0,
            "avg_output_length": sum(r.get('output_length', 0) for r in successful) / len(successful) if successful else 0,
            "results": results
        }
        
        # Add accuracy metrics if available
        with_accuracy = [r for r in successful if 'accuracy_score' in r]
        if with_accuracy:
            evaluation["avg_accuracy"] = sum(r['accuracy_score'] for r in with_accuracy) / len(with_accuracy)
            evaluation["avg_relevance"] = sum(r['relevance_score'] for r in with_accuracy) / len(with_accuracy)
        
        self.evaluations.append(evaluation)
        
        logger.info(f"✓ Evaluation complete: {len(successful)}/{len(test_cases)} successful")
        
        return evaluation
    
    def compare_temperatures(self, llm_client, task: str, test_cases: List[Dict],
                           temperatures: List[float]) -> Dict[str, Any]:
        """
        Compare performance across different temperatures.
        
        Args:
            llm_client: LLM client
            task: Task name
            test_cases: Test cases
            temperatures: List of temperatures to test
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing temperatures: {temperatures}")
        
        comparison = {
            "task": task,
            "temperatures": temperatures,
            "evaluations": {}
        }
        
        for temp in temperatures:
            config = {"temperature": temp, "max_tokens": 1000}
            evaluation = self.evaluate_model_config(llm_client, task, test_cases, config)
            comparison["evaluations"][str(temp)] = evaluation
        
        # Find best temperature
        best_temp = max(comparison["evaluations"].items(),
                       key=lambda x: x[1]['success_rate'])
        comparison["best_temperature"] = {
            "value": float(best_temp[0]),
            "success_rate": best_temp[1]['success_rate']
        }
        
        return comparison
    
    def compare_max_tokens(self, llm_client, task: str, test_cases: List[Dict],
                          token_limits: List[int]) -> Dict[str, Any]:
        """
        Compare performance across different token limits.
        
        Args:
            llm_client: LLM client
            task: Task name
            test_cases: Test cases
            token_limits: List of max_tokens to test
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing max_tokens: {token_limits}")
        
        comparison = {
            "task": task,
            "token_limits": token_limits,
            "evaluations": {}
        }
        
        for limit in token_limits:
            config = {"temperature": 0.7, "max_tokens": limit}
            evaluation = self.evaluate_model_config(llm_client, task, test_cases, config)
            comparison["evaluations"][str(limit)] = evaluation
        
        return comparison
    
    def _calculate_accuracy(self, output: str, expected: Any) -> float:
        """
        Calculate accuracy score.
        
        Args:
            output: Generated output
            expected: Expected output
            
        Returns:
            Accuracy score (0-1)
        """
        # Simple token-based similarity
        output_tokens = set(output.lower().split())
        expected_str = str(expected) if not isinstance(expected, str) else expected
        expected_tokens = set(expected_str.lower().split())
        
        if not expected_tokens:
            return 0.0
        
        overlap = len(output_tokens & expected_tokens)
        union = len(output_tokens | expected_tokens)
        
        return overlap / union if union > 0 else 0.0
    
    def _calculate_relevance(self, output: str, expected: Any) -> float:
        """
        Calculate relevance score.
        
        Args:
            output: Generated output
            expected: Expected output
            
        Returns:
            Relevance score (0-1)
        """
        # Check if output contains expected keywords
        expected_str = str(expected).lower()
        output_lower = output.lower()
        
        # Extract important words (length > 4)
        expected_keywords = [w for w in expected_str.split() if len(w) > 4]
        
        if not expected_keywords:
            return 0.5  # Neutral score if no keywords
        
        found_keywords = sum(1 for kw in expected_keywords if kw in output_lower)
        
        return found_keywords / len(expected_keywords)
    
    def export_comparison(self, comparison_name: str, comparison_data: Dict) -> str:
        """
        Export comparison results.
        
        Args:
            comparison_name: Name for the comparison
            comparison_data: Comparison data
            
        Returns:
            Path to exported file
        """
        filepath = self.results_dir / f"{comparison_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filepath, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        logger.info(f"✓ Comparison exported to {filepath}")
        return str(filepath)
    
    def generate_comparison_report(self, comparison_data: Dict) -> str:
        """
        Generate readable comparison report.
        
        Args:
            comparison_data: Comparison data
            
        Returns:
            Report text
        """
        report = f"""
{'='*70}
MODEL CONFIGURATION COMPARISON REPORT
Task: {comparison_data.get('task', 'Unknown')}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

"""
        
        evaluations = comparison_data.get('evaluations', {})
        
        if 'temperatures' in comparison_data:
            report += "TEMPERATURE COMPARISON:\n"
            for temp, eval_data in evaluations.items():
                report += f"\n  Temperature: {temp}\n"
                report += f"    - Success Rate: {eval_data['success_rate']:.1%}\n"
                report += f"    - Avg Execution Time: {eval_data['avg_execution_time']:.3f}s\n"
                if 'avg_accuracy' in eval_data:
                    report += f"    - Avg Accuracy: {eval_data['avg_accuracy']:.2f}\n"
            
            best = comparison_data.get('best_temperature', {})
            report += f"\n  Best Temperature: {best.get('value')} (Success Rate: {best.get('success_rate', 0):.1%})\n"
        
        elif 'token_limits' in comparison_data:
            report += "TOKEN LIMIT COMPARISON:\n"
            for limit, eval_data in evaluations.items():
                report += f"\n  Max Tokens: {limit}\n"
                report += f"    - Success Rate: {eval_data['success_rate']:.1%}\n"
                report += f"    - Avg Output Length: {eval_data['avg_output_length']:.0f} chars\n"
                report += f"    - Avg Execution Time: {eval_data['avg_execution_time']:.3f}s\n"
        
        report += f"\n{'='*70}\n"
        
        return report
