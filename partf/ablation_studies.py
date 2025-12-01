"""
Ablation Studies for Prompt Components
Systematic removal of prompt components to measure their impact
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


class AblationStudies:
    """
    Conduct ablation studies on prompt components:
    - Remove instructions
    - Remove examples
    - Remove context
    - Remove structure
    - Remove reasoning steps
    """
    
    def __init__(self, llm_client, results_dir: str = "partf/experiments/results"):
        """
        Initialize ablation studies.
        
        Args:
            llm_client: LLM client
            results_dir: Directory for results
        """
        self.llm = llm_client
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        self.studies: List[Dict] = []
        
        logger.info("✓ Ablation Studies initialized")
    
    def study_instruction_removal(self, base_prompt: str, test_cases: List[Dict],
                                  instruction_markers: List[str] = None) -> Dict[str, Any]:
        """
        Study impact of removing instructions from prompt.
        
        Args:
            base_prompt: Full prompt with instructions
            test_cases: Test cases
            instruction_markers: Markers that indicate instructions (e.g., "Guidelines:", "Instructions:")
            
        Returns:
            Ablation study results
        """
        logger.info("Ablation Study: Instruction Removal")
        
        if instruction_markers is None:
            instruction_markers = ["Instructions:", "Guidelines:", "Rules:", "Requirements:"]
        
        # Test baseline (full prompt)
        baseline = self._test_prompt_variant("baseline", base_prompt, test_cases)
        
        # Create ablated prompt (remove instructions)
        ablated_prompt = base_prompt
        for marker in instruction_markers:
            if marker in ablated_prompt:
                # Remove from marker to next section or end
                start = ablated_prompt.find(marker)
                # Find next double newline or end
                end = ablated_prompt.find("\n\n", start)
                if end == -1:
                    end = len(ablated_prompt)
                ablated_prompt = ablated_prompt[:start] + ablated_prompt[end:]
        
        # Test ablated prompt
        ablated = self._test_prompt_variant("without_instructions", ablated_prompt, test_cases)
        
        # Compare results
        study_results = {
            "study_type": "instruction_removal",
            "baseline": baseline,
            "ablated": ablated,
            "impact": {
                "success_rate_change": ablated["success_rate"] - baseline["success_rate"],
                "execution_time_change": ablated["avg_execution_time"] - baseline["avg_execution_time"],
                "quality_degradation": baseline["success_rate"] - ablated["success_rate"]
            },
            "conclusion": self._generate_conclusion("instructions", baseline, ablated)
        }
        
        self.studies.append(study_results)
        
        logger.info(f"  Impact: {study_results['impact']['quality_degradation']:.1%} quality degradation")
        
        return study_results
    
    def study_example_removal(self, base_prompt_with_examples: str, 
                             test_cases: List[Dict],
                             example_marker: str = "Example") -> Dict[str, Any]:
        """
        Study impact of removing few-shot examples.
        
        Args:
            base_prompt_with_examples: Prompt with few-shot examples
            test_cases: Test cases
            example_marker: Marker for examples
            
        Returns:
            Ablation study results
        """
        logger.info("Ablation Study: Example Removal")
        
        # Test baseline (with examples)
        baseline = self._test_prompt_variant("with_examples", base_prompt_with_examples, test_cases)
        
        # Create ablated prompt (remove examples)
        ablated_prompt = base_prompt_with_examples
        lines = ablated_prompt.split('\n')
        filtered_lines = []
        skip_until_blank = False
        
        for line in lines:
            if example_marker in line:
                skip_until_blank = True
                continue
            
            if skip_until_blank:
                if line.strip() == "":
                    skip_until_blank = False
                continue
            
            filtered_lines.append(line)
        
        ablated_prompt = '\n'.join(filtered_lines)
        
        # Test ablated prompt (zero-shot)
        ablated = self._test_prompt_variant("without_examples", ablated_prompt, test_cases)
        
        study_results = {
            "study_type": "example_removal",
            "baseline": baseline,
            "ablated": ablated,
            "impact": {
                "success_rate_change": ablated["success_rate"] - baseline["success_rate"],
                "execution_time_change": ablated["avg_execution_time"] - baseline["avg_execution_time"],
                "quality_degradation": baseline["success_rate"] - ablated["success_rate"]
            },
            "conclusion": self._generate_conclusion("examples", baseline, ablated)
        }
        
        self.studies.append(study_results)
        
        logger.info(f"  Impact: {study_results['impact']['quality_degradation']:.1%} quality degradation")
        
        return study_results
    
    def study_context_removal(self, base_prompt: str, test_cases: List[Dict],
                             context_marker: str = "Context:") -> Dict[str, Any]:
        """
        Study impact of removing context.
        
        Args:
            base_prompt: Prompt with context
            test_cases: Test cases
            context_marker: Marker for context section
            
        Returns:
            Ablation study results
        """
        logger.info("Ablation Study: Context Removal")
        
        # Test baseline (with context)
        baseline = self._test_prompt_variant("with_context", base_prompt, test_cases)
        
        # Create ablated prompt (remove context)
        ablated_prompt = base_prompt
        if context_marker in ablated_prompt:
            start = ablated_prompt.find(context_marker)
            end = ablated_prompt.find("\n\n", start)
            if end == -1:
                end = ablated_prompt.find("\n", start + len(context_marker)) + 1
            ablated_prompt = ablated_prompt[:start] + ablated_prompt[end:]
        
        # Test ablated prompt
        ablated = self._test_prompt_variant("without_context", ablated_prompt, test_cases)
        
        study_results = {
            "study_type": "context_removal",
            "baseline": baseline,
            "ablated": ablated,
            "impact": {
                "success_rate_change": ablated["success_rate"] - baseline["success_rate"],
                "execution_time_change": ablated["avg_execution_time"] - baseline["avg_execution_time"],
                "quality_degradation": baseline["success_rate"] - ablated["success_rate"]
            },
            "conclusion": self._generate_conclusion("context", baseline, ablated)
        }
        
        self.studies.append(study_results)
        
        logger.info(f"  Impact: {study_results['impact']['quality_degradation']:.1%} quality degradation")
        
        return study_results
    
    def study_structure_removal(self, structured_prompt: str, test_cases: List[Dict]) -> Dict[str, Any]:
        """
        Study impact of removing structural formatting.
        
        Args:
            structured_prompt: Prompt with structure (bullets, numbering, sections)
            test_cases: Test cases
            
        Returns:
            Ablation study results
        """
        logger.info("Ablation Study: Structure Removal")
        
        # Test baseline (structured)
        baseline = self._test_prompt_variant("structured", structured_prompt, test_cases)
        
        # Create unstructured prompt (remove formatting)
        unstructured_prompt = structured_prompt
        # Remove bullets
        unstructured_prompt = unstructured_prompt.replace('- ', '')
        unstructured_prompt = unstructured_prompt.replace('* ', '')
        # Remove numbering
        import re
        unstructured_prompt = re.sub(r'\d+\.\s+', '', unstructured_prompt)
        # Remove section markers
        unstructured_prompt = unstructured_prompt.replace('##', '')
        unstructured_prompt = unstructured_prompt.replace('**', '')
        
        # Test unstructured prompt
        ablated = self._test_prompt_variant("unstructured", unstructured_prompt, test_cases)
        
        study_results = {
            "study_type": "structure_removal",
            "baseline": baseline,
            "ablated": ablated,
            "impact": {
                "success_rate_change": ablated["success_rate"] - baseline["success_rate"],
                "execution_time_change": ablated["avg_execution_time"] - baseline["avg_execution_time"],
                "quality_degradation": baseline["success_rate"] - ablated["success_rate"]
            },
            "conclusion": self._generate_conclusion("structure", baseline, ablated)
        }
        
        self.studies.append(study_results)
        
        logger.info(f"  Impact: {study_results['impact']['quality_degradation']:.1%} quality degradation")
        
        return study_results
    
    def study_reasoning_steps_removal(self, cot_prompt: str, test_cases: List[Dict]) -> Dict[str, Any]:
        """
        Study impact of removing chain-of-thought reasoning steps.
        
        Args:
            cot_prompt: Prompt with reasoning steps
            test_cases: Test cases
            
        Returns:
            Ablation study results
        """
        logger.info("Ablation Study: Reasoning Steps Removal")
        
        # Test baseline (with reasoning steps)
        baseline = self._test_prompt_variant("with_reasoning", cot_prompt, test_cases)
        
        # Create ablated prompt (direct answer request)
        ablated_prompt = cot_prompt
        reasoning_markers = [
            "step by step",
            "Let's think",
            "reasoning:",
            "First,",
            "Then,",
            "Finally,"
        ]
        
        for marker in reasoning_markers:
            if marker in ablated_prompt.lower():
                # Remove reasoning instruction
                ablated_prompt = ablated_prompt.replace(marker, '')
                ablated_prompt = ablated_prompt.replace(marker.title(), '')
        
        # Test ablated prompt
        ablated = self._test_prompt_variant("without_reasoning", ablated_prompt, test_cases)
        
        study_results = {
            "study_type": "reasoning_steps_removal",
            "baseline": baseline,
            "ablated": ablated,
            "impact": {
                "success_rate_change": ablated["success_rate"] - baseline["success_rate"],
                "execution_time_change": ablated["avg_execution_time"] - baseline["avg_execution_time"],
                "quality_degradation": baseline["success_rate"] - ablated["success_rate"]
            },
            "conclusion": self._generate_conclusion("reasoning steps", baseline, ablated)
        }
        
        self.studies.append(study_results)
        
        logger.info(f"  Impact: {study_results['impact']['quality_degradation']:.1%} quality degradation")
        
        return study_results
    
    def comprehensive_ablation(self, full_prompt: str, test_cases: List[Dict],
                              components: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive ablation study removing multiple components.
        
        Args:
            full_prompt: Full featured prompt
            test_cases: Test cases
            components: Components to ablate
            
        Returns:
            Comprehensive ablation results
        """
        logger.info("Comprehensive Ablation Study")
        
        if components is None:
            components = ["instructions", "examples", "context", "structure", "reasoning"]
        
        # Test baseline (full prompt)
        logger.info("  Testing baseline (full prompt)")
        baseline = self._test_prompt_variant("full_prompt", full_prompt, test_cases)
        
        ablation_results = {
            "baseline": baseline,
            "ablations": {}
        }
        
        # Test each component removal
        for component in components:
            logger.info(f"  Testing without: {component}")
            
            # Create ablated variant (this is simplified - real implementation would be more sophisticated)
            if component == "instructions":
                study = self.study_instruction_removal(full_prompt, test_cases)
            elif component == "examples":
                study = self.study_example_removal(full_prompt, test_cases)
            elif component == "context":
                study = self.study_context_removal(full_prompt, test_cases)
            elif component == "structure":
                study = self.study_structure_removal(full_prompt, test_cases)
            elif component == "reasoning":
                study = self.study_reasoning_steps_removal(full_prompt, test_cases)
            
            ablation_results["ablations"][component] = study
        
        # Rank component importance
        importance_ranking = sorted(
            ablation_results["ablations"].items(),
            key=lambda x: x[1]["impact"]["quality_degradation"],
            reverse=True
        )
        
        ablation_results["component_importance"] = [
            {
                "component": comp,
                "importance_score": data["impact"]["quality_degradation"]
            }
            for comp, data in importance_ranking
        ]
        
        return ablation_results
    
    def _test_prompt_variant(self, variant_name: str, prompt: str, 
                            test_cases: List[Dict]) -> Dict[str, Any]:
        """
        Test a prompt variant on test cases.
        
        Args:
            variant_name: Name of the variant
            prompt: Prompt text
            test_cases: Test cases
            
        Returns:
            Test results
        """
        results = []
        
        for test_case in test_cases:
            try:
                # Format prompt with test input
                formatted_prompt = prompt.format(**test_case.get('input', {}))
                
                # Generate
                start_time = time.time()
                output = self.llm.generate(formatted_prompt, temperature=0.7, max_tokens=1000)
                execution_time = time.time() - start_time
                
                # Evaluate
                expected = test_case.get('expected_output')
                success = len(output) > 0
                
                result = {
                    "success": success,
                    "output": output,
                    "execution_time": execution_time
                }
                
                if expected:
                    result["match_score"] = self._calculate_match(output, expected)
                
                results.append(result)
            
            except Exception as e:
                results.append({
                    "success": False,
                    "error": str(e)
                })
        
        # Calculate metrics
        successful = [r for r in results if r.get('success')]
        
        return {
            "variant_name": variant_name,
            "num_tests": len(test_cases),
            "num_successful": len(successful),
            "success_rate": len(successful) / len(test_cases) if test_cases else 0,
            "avg_execution_time": sum(r.get('execution_time', 0) for r in successful) / len(successful) if successful else 0,
            "results": results
        }
    
    def _calculate_match(self, output: str, expected: Any) -> float:
        """Calculate match score between output and expected."""
        output_tokens = set(output.lower().split())
        expected_tokens = set(str(expected).lower().split())
        
        if not expected_tokens:
            return 0.0
        
        overlap = len(output_tokens & expected_tokens)
        return overlap / len(expected_tokens)
    
    def _generate_conclusion(self, component: str, baseline: Dict, ablated: Dict) -> str:
        """Generate conclusion about component importance."""
        degradation = baseline["success_rate"] - ablated["success_rate"]
        
        if degradation > 0.3:
            importance = "CRITICAL"
            description = f"Removing {component} causes severe performance degradation ({degradation:.1%})"
        elif degradation > 0.15:
            importance = "HIGH"
            description = f"Removing {component} significantly reduces performance ({degradation:.1%})"
        elif degradation > 0.05:
            importance = "MEDIUM"
            description = f"Removing {component} moderately affects performance ({degradation:.1%})"
        else:
            importance = "LOW"
            description = f"Removing {component} has minimal impact ({degradation:.1%})"
        
        return f"{importance}: {description}"
    
    def export_studies(self, filename: Optional[str] = None) -> str:
        """Export ablation studies."""
        if not filename:
            filename = f"ablation_studies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.studies, f, indent=2)
        
        logger.info(f"✓ Ablation studies exported to {filepath}")
        return str(filepath)
    
    def generate_report(self) -> str:
        """Generate ablation studies report."""
        report = f"""
{'='*70}
ABLATION STUDIES REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

Total Studies: {len(self.studies)}

"""
        
        for i, study in enumerate(self.studies, 1):
            report += f"""
STUDY {i}: {study['study_type'].replace('_', ' ').title()}
{'='*70}

Baseline Performance:
  - Success Rate: {study['baseline']['success_rate']:.1%}
  - Avg Time: {study['baseline']['avg_execution_time']:.3f}s

Ablated Performance:
  - Success Rate: {study['ablated']['success_rate']:.1%}
  - Avg Time: {study['ablated']['avg_execution_time']:.3f}s

Impact:
  - Quality Degradation: {study['impact']['quality_degradation']:.1%}
  - Time Change: {study['impact']['execution_time_change']:.3f}s

Conclusion:
  {study['conclusion']}

"""
        
        report += f"{'='*70}\n"
        
        return report
