"""
Prompt Engineering and Optimization
Systematic prompt design, testing, and iteration
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from datetime import datetime
import time

from partf.prompt_library import PromptLibrary
from partf.few_shot_examples import FewShotLibrary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptEngineer:
    """
    Prompt engineering system with:
    - Zero-shot, few-shot, and chain-of-thought prompting
    - Iterative refinement based on results
    - Performance tracking
    - Ablation studies
    """
    
    def __init__(self, llm_client, prompt_library: Optional[PromptLibrary] = None,
                 few_shot_library: Optional[FewShotLibrary] = None):
        """
        Initialize prompt engineer.
        
        Args:
            llm_client: LLM client for testing prompts
            prompt_library: Prompt library (creates new if None)
            few_shot_library: Few-shot library (creates new if None)
        """
        self.llm = llm_client
        self.prompt_lib = prompt_library or PromptLibrary()
        self.few_shot_lib = few_shot_library or FewShotLibrary()
        
        # Experimentation tracking
        self.experiments: List[Dict] = []
        self.iterations: Dict[str, List[Dict]] = {}
        
        logger.info("✓ Prompt Engineer initialized")
    
    def test_prompt(self, task: str, variant: str, test_input: Dict[str, Any],
                   expected_output: Optional[Any] = None,
                   temperature: float = 0.7,
                   max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Test a specific prompt variant.
        
        Args:
            task: Task type
            variant: Prompt variant
            test_input: Input parameters for the prompt
            expected_output: Expected output (for comparison)
            temperature: LLM temperature
            max_tokens: Max tokens to generate
            
        Returns:
            Test results with metrics
        """
        logger.info(f"Testing prompt: {task}/{variant}")
        
        # Get prompt info to check required parameters
        try:
            prompt_info = self.prompt_lib.get_prompt_info(task, variant)
            required_params = prompt_info.get("parameters", [])
            
            # Prepare parameters with defaults/extractions
            params = test_input.copy()
            
            # Auto-extract missing parameters
            if "topic" in required_params and "topic" not in params:
                # Extract topic from question if available
                if "question" in params:
                    # Simple extraction: use first few words or a default
                    question = params.get("question", "")
                    # Use "sustainability" as default topic for environmental questions
                    params["topic"] = "sustainability and environment"
                else:
                    params["topic"] = "sustainability and environment"
            
            if "context" in required_params and "context" not in params:
                # Provide default context for environmental questions
                if "question" in params:
                    params["context"] = "Environmental sustainability and climate change"
                else:
                    params["context"] = "General environmental knowledge"
            
            if "length" in required_params and "length" not in params:
                # Extract from num_sentences if available
                if "num_sentences" in params:
                    params["length"] = f"{params['num_sentences']} sentences"
                else:
                    params["length"] = "brief"
            
            # Get prompt with prepared parameters
            prompt = self.prompt_lib.get_prompt(task, variant, **params)
        except Exception as e:
            return {
                "success": False,
                "error": f"Prompt generation failed: {e}"
            }
        
        # Execute
        start_time = time.time()
        try:
            output = self.llm.generate(prompt, temperature=temperature, max_tokens=max_tokens)
            execution_time = time.time() - start_time
            
            result = {
                "success": True,
                "task": task,
                "variant": variant,
                "prompt": prompt,
                "output": output,
                "execution_time": execution_time,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timestamp": datetime.now().isoformat()
            }
            
            # Compare with expected output if provided
            if expected_output:
                result["match_score"] = self._calculate_match_score(output, expected_output)
            
            # Log experiment
            self.experiments.append(result)
            
            return result
        
        except Exception as e:
            logger.error(f"Prompt execution failed: {e}")
            return {
                "success": False,
                "task": task,
                "variant": variant,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def compare_variants(self, task: str, variants: List[str], 
                        test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple prompt variants on same test cases.
        
        Args:
            task: Task type
            variants: List of variant names
            test_cases: List of test inputs
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(variants)} variants on {len(test_cases)} test cases")
        
        results = {
            "task": task,
            "variants": variants,
            "test_cases": test_cases,
            "variant_results": {},
            "summary": {}
        }
        
        # Test each variant on all test cases
        for variant in variants:
            variant_results = []
            
            for i, test_case in enumerate(test_cases):
                logger.info(f"  Variant '{variant}' - Test case {i+1}/{len(test_cases)}")
                
                result = self.test_prompt(
                    task=task,
                    variant=variant,
                    test_input=test_case.get('input', test_case),
                    expected_output=test_case.get('expected_output')
                )
                
                variant_results.append(result)
            
            results["variant_results"][variant] = variant_results
            
            # Calculate summary statistics
            successful = [r for r in variant_results if r.get('success')]
            results["summary"][variant] = {
                "success_rate": len(successful) / len(variant_results) if variant_results else 0,
                "avg_execution_time": sum(r.get('execution_time', 0) for r in successful) / len(successful) if successful else 0,
                "avg_match_score": sum(r.get('match_score', 0) for r in successful) / len(successful) if successful else 0
            }
        
        # Determine best variant
        best_variant = max(results["summary"].items(), 
                          key=lambda x: (x[1]["success_rate"], x[1]["avg_match_score"]))
        results["best_variant"] = {
            "name": best_variant[0],
            "metrics": best_variant[1]
        }
        
        logger.info(f"✓ Best variant: {best_variant[0]}")
        
        return results
    
    def test_few_shot(self, task: str, num_shots: int, test_input: Any,
                     expected_output: Optional[Any] = None) -> Dict[str, Any]:
        """
        Test few-shot prompting with different numbers of examples.
        
        Args:
            task: Task type
            num_shots: Number of examples to include
            test_input: Input to process
            expected_output: Expected output
            
        Returns:
            Test results
        """
        logger.info(f"Testing {num_shots}-shot prompting for {task}")
        
        # Create few-shot prompt
        prompt = self.few_shot_lib.format_few_shot_prompt(task, num_shots, test_input)
        
        # Execute
        start_time = time.time()
        try:
            output = self.llm.generate(prompt, temperature=0.7, max_tokens=1000)
            execution_time = time.time() - start_time
            
            result = {
                "success": True,
                "task": task,
                "num_shots": num_shots,
                "prompt": prompt,
                "output": output,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
            
            if expected_output:
                result["match_score"] = self._calculate_match_score(output, expected_output)
            
            return result
        
        except Exception as e:
            return {
                "success": False,
                "task": task,
                "num_shots": num_shots,
                "error": str(e)
            }
    
    def ablation_study(self, task: str, base_variant: str,
                      test_cases: List[Dict], ablations: List[str]) -> Dict[str, Any]:
        """
        Perform ablation study by removing components from prompt.
        
        Args:
            task: Task type
            base_variant: Base prompt variant
            test_cases: Test cases
            ablations: Components to ablate (e.g., "examples", "instructions")
            
        Returns:
            Ablation study results
        """
        logger.info(f"Running ablation study on {task}/{base_variant}")
        
        results = {
            "task": task,
            "base_variant": base_variant,
            "ablations": {},
            "baseline": None
        }
        
        # Test baseline (full prompt)
        logger.info("  Testing baseline (full prompt)")
        baseline_results = []
        for test_case in test_cases:
            result = self.test_prompt(task, base_variant, test_case['input'])
            baseline_results.append(result)
        
        results["baseline"] = {
            "results": baseline_results,
            "success_rate": sum(1 for r in baseline_results if r.get('success')) / len(baseline_results),
            "avg_time": sum(r.get('execution_time', 0) for r in baseline_results) / len(baseline_results)
        }
        
        # Test each ablation
        for ablation in ablations:
            logger.info(f"  Testing ablation: remove {ablation}")
            # This would require creating modified prompts
            # For now, log that ablation testing is needed
            results["ablations"][ablation] = {
                "note": "Ablation testing requires modified prompt variants"
            }
        
        return results
    
    def iterative_refinement(self, task: str, initial_variant: str,
                           test_cases: List[Dict], max_iterations: int = 3) -> Dict[str, Any]:
        """
        Iteratively refine prompt based on results.
        
        Args:
            task: Task type
            initial_variant: Starting prompt variant
            test_cases: Test cases for evaluation
            max_iterations: Maximum refinement iterations
            
        Returns:
            Refinement history and final prompt
        """
        logger.info(f"Starting iterative refinement for {task}/{initial_variant}")
        
        refinement_history = []
        current_variant = initial_variant
        
        for iteration in range(max_iterations):
            logger.info(f"  Iteration {iteration + 1}/{max_iterations}")
            
            # Test current variant
            results = []
            for test_case in test_cases:
                result = self.test_prompt(task, current_variant, test_case['input'],
                                        test_case.get('expected_output'))
                results.append(result)
            
            # Analyze results
            success_rate = sum(1 for r in results if r.get('success')) / len(results)
            avg_match = sum(r.get('match_score', 0) for r in results) / len(results)
            
            iteration_data = {
                "iteration": iteration + 1,
                "variant": current_variant,
                "success_rate": success_rate,
                "avg_match_score": avg_match,
                "results": results
            }
            
            refinement_history.append(iteration_data)
            
            # Check if refinement needed
            if success_rate >= 0.9 and avg_match >= 0.8:
                logger.info(f"  ✓ Good performance achieved (success: {success_rate:.1%}, match: {avg_match:.2f})")
                break
            
            # Analyze failures and suggest refinements
            failures = [r for r in results if not r.get('success') or r.get('match_score', 0) < 0.7]
            
            if failures:
                logger.info(f"  ⚠ {len(failures)} failures detected. Suggesting refinements...")
                refinement_suggestions = self._analyze_failures(failures)
                iteration_data["refinement_suggestions"] = refinement_suggestions
        
        return {
            "task": task,
            "initial_variant": initial_variant,
            "refinement_history": refinement_history,
            "final_performance": {
                "success_rate": refinement_history[-1]["success_rate"],
                "avg_match_score": refinement_history[-1]["avg_match_score"]
            }
        }
    
    def _calculate_match_score(self, output: str, expected: Any) -> float:
        """
        Calculate similarity between output and expected result.
        Simple implementation - can be enhanced with semantic similarity.
        
        Args:
            output: Generated output
            expected: Expected output
            
        Returns:
            Match score (0-1)
        """
        if isinstance(expected, str):
            # Simple token overlap for text
            output_tokens = set(output.lower().split())
            expected_tokens = set(str(expected).lower().split())
            
            if not expected_tokens:
                return 0.0
            
            overlap = len(output_tokens & expected_tokens)
            return overlap / len(expected_tokens)
        
        elif isinstance(expected, dict):
            # For structured output, check key presence
            try:
                # Try to parse output as JSON
                import json
                output_dict = json.loads(output)
                expected_keys = set(expected.keys())
                output_keys = set(output_dict.keys())
                overlap = len(expected_keys & output_keys)
                return overlap / len(expected_keys)
            except:
                return 0.0
        
        return 0.0
    
    def _analyze_failures(self, failures: List[Dict]) -> List[str]:
        """
        Analyze failure patterns and suggest improvements.
        
        Args:
            failures: List of failed results
            
        Returns:
            List of refinement suggestions
        """
        suggestions = []
        
        # Check for common issues
        for failure in failures:
            error = failure.get('error', '')
            output = failure.get('output', '')
            
            if 'json' in error.lower() or 'parse' in error.lower():
                suggestions.append("Add explicit JSON format instructions with example structure")
            
            if len(output) < 50:
                suggestions.append("Output too short - add instruction to be more detailed")
            
            if 'timeout' in error.lower():
                suggestions.append("Reduce prompt complexity or max_tokens")
        
        # Remove duplicates
        suggestions = list(set(suggestions))
        
        if not suggestions:
            suggestions.append("Review prompt clarity and add more specific instructions")
        
        return suggestions
    
    def export_experiments(self, filepath: Optional[str] = None) -> str:
        """
        Export all experiment results.
        
        Args:
            filepath: Output filepath
            
        Returns:
            Path to exported file
        """
        if not filepath:
            filepath = f"partf/experiments/results/experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = Path(filepath)
        filepath.parent.mkdir(exist_ok=True, parents=True)
        
        export_data = {
            "total_experiments": len(self.experiments),
            "experiments": self.experiments,
            "summary": {
                "success_rate": sum(1 for e in self.experiments if e.get('success')) / len(self.experiments) if self.experiments else 0,
                "avg_execution_time": sum(e.get('execution_time', 0) for e in self.experiments) / len(self.experiments) if self.experiments else 0
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"✓ Exported {len(self.experiments)} experiments to {filepath}")
        return str(filepath)
    
    def generate_report(self) -> str:
        """
        Generate human-readable report of experiments.
        
        Returns:
            Report text
        """
        if not self.experiments:
            return "No experiments to report."
        
        report = f"""
{'='*70}
PROMPT ENGINEERING EXPERIMENT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

SUMMARY:
- Total Experiments: {len(self.experiments)}
- Successful: {sum(1 for e in self.experiments if e.get('success'))}
- Failed: {sum(1 for e in self.experiments if not e.get('success'))}
- Success Rate: {sum(1 for e in self.experiments if e.get('success')) / len(self.experiments):.1%}

PERFORMANCE METRICS:
"""
        
        # Group by task
        by_task = {}
        for exp in self.experiments:
            task = exp.get('task', 'unknown')
            if task not in by_task:
                by_task[task] = []
            by_task[task].append(exp)
        
        for task, exps in by_task.items():
            successful = [e for e in exps if e.get('success')]
            report += f"\n{task.upper()}:\n"
            report += f"  - Experiments: {len(exps)}\n"
            report += f"  - Success Rate: {len(successful) / len(exps):.1%}\n"
            
            if successful:
                avg_time = sum(e.get('execution_time', 0) for e in successful) / len(successful)
                report += f"  - Avg Execution Time: {avg_time:.3f}s\n"
                
                # Variant performance
                by_variant = {}
                for e in successful:
                    variant = e.get('variant', 'unknown')
                    if variant not in by_variant:
                        by_variant[variant] = []
                    by_variant[variant].append(e)
                
                report += f"  - Variants Tested:\n"
                for variant, v_exps in by_variant.items():
                    report += f"    - {variant}: {len(v_exps)} experiments\n"
        
        report += f"\n{'='*70}\n"
        
        return report
        for exp in self.experiments:
            task = exp.get('task', 'unknown')
            if task not in by_task:
                by_task[task] = []
            by_task[task].append(exp)
        
        for task, exps in by_task.items():
            successful = [e for e in exps if e.get('success')]
            report += f"\n{task.upper()}:\n"
            report += f"  - Experiments: {len(exps)}\n"
            report += f"  - Success Rate: {len(successful) / len(exps):.1%}\n"
            
            if successful:
                avg_time = sum(e.get('execution_time', 0) for e in successful) / len(successful)
                report += f"  - Avg Execution Time: {avg_time:.3f}s\n"
                
                # Variant performance
                by_variant = {}
                for e in successful:
                    variant = e.get('variant', 'unknown')
                    if variant not in by_variant:
                        by_variant[variant] = []
                    by_variant[variant].append(e)
                
                report += f"  - Variants Tested:\n"
                for variant, v_exps in by_variant.items():
                    report += f"    - {variant}: {len(v_exps)} experiments\n"
        
        report += f"\n{'='*70}\n"
        
        return report

