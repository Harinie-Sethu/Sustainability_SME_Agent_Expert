"""
Failure Analysis and Documentation
Systematically analyze and document failure cases
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FailureAnalyzer:
    """
    Analyze and document failure cases:
    - Categorize failure types
    - Identify patterns
    - Document root causes
    - Suggest fixes
    """
    
    def __init__(self, results_dir: str = "partf/experiments/results"):
        """
        Initialize failure analyzer.
        
        Args:
            results_dir: Directory for results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        self.failure_cases: List[Dict] = []
        self.failure_patterns: Dict[str, List[Dict]] = {}
        
        logger.info(" Failure Analyzer initialized")
    
    def record_failure(self, task: str, prompt: str, input_data: Dict,
                      output: str, expected_output: Any, error: Optional[str] = None):
        """
        Record a failure case.
        
        Args:
            task: Task name
            prompt: Prompt used
            input_data: Input data
            output: Generated output
            expected_output: Expected output
            error: Error message if any
        """
        failure_case = {
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "prompt": prompt,
            "input": input_data,
            "output": output,
            "expected_output": expected_output,
            "error": error,
            "failure_type": self._classify_failure(output, expected_output, error)
        }
        
        self.failure_cases.append(failure_case)
        
        # Add to pattern tracking
        failure_type = failure_case["failure_type"]
        if failure_type not in self.failure_patterns:
            self.failure_patterns[failure_type] = []
        self.failure_patterns[failure_type].append(failure_case)
        
        logger.info(f"Recorded failure: {failure_type}")
    
    def _classify_failure(self, output: str, expected: Any, error: Optional[str]) -> str:
        """
        Classify the type of failure.
        
        Args:
            output: Generated output
            expected: Expected output
            error: Error message
            
        Returns:
            Failure classification
        """
        if error:
            if 'timeout' in error.lower():
                return "timeout"
            elif 'json' in error.lower() or 'parse' in error.lower():
                return "format_error"
            elif 'token' in error.lower():
                return "token_limit"
            else:
                return "execution_error"
        
        # No error but incorrect output
        if not output or len(output.strip()) < 10:
            return "empty_output"
        
        # Check if output is off-topic
        expected_str = str(expected).lower()
        output_lower = output.lower()
        
        # Extract key terms from expected
        expected_words = [w for w in expected_str.split() if len(w) > 4]
        matches = sum(1 for w in expected_words if w in output_lower)
        
        if len(expected_words) > 0:
            match_ratio = matches / len(expected_words)
            if match_ratio < 0.2:
                return "off_topic"
            elif match_ratio < 0.5:
                return "partial_correctness"
        
        # Check for hallucination indicators
        if any(phrase in output_lower for phrase in ['as an ai', 'i cannot', 'i apologize']):
            return "refusal"
        
        return "incorrect_content"
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """
        Analyze failure patterns.
        
        Returns:
            Analysis results
        """
        logger.info("Analyzing failure patterns...")
        
        if not self.failure_cases:
            return {"message": "No failures to analyze"}
        
        analysis = {
            "total_failures": len(self.failure_cases),
            "failure_distribution": {},
            "common_issues": [],
            "task_failures": {},
            "temporal_patterns": {}
        }
        
        # Failure type distribution
        failure_types = Counter(f["failure_type"] for f in self.failure_cases)
        analysis["failure_distribution"] = dict(failure_types.most_common())
        
        # Task-specific failures
        task_failures = {}
        for failure in self.failure_cases:
            task = failure["task"]
            if task not in task_failures:
                task_failures[task] = []
            task_failures[task].append(failure)
        
        analysis["task_failures"] = {
            task: len(failures)
            for task, failures in task_failures.items()
        }
        
        # Identify common issues
        analysis["common_issues"] = self._identify_common_issues()
        
        return analysis
    
    def _identify_common_issues(self) -> List[Dict]:
        """Identify common issues across failures."""
        issues = []
        
        # Check for format errors
        format_failures = self.failure_patterns.get("format_error", [])
        if len(format_failures) > len(self.failure_cases) * 0.2:
            issues.append({
                "issue": "Frequent format errors",
                "frequency": len(format_failures),
                "severity": "HIGH",
                "suggestion": "Add explicit format instructions with examples. Use JSON schema."
            })
        
        # Check for empty outputs
        empty_failures = self.failure_patterns.get("empty_output", [])
        if len(empty_failures) > len(self.failure_cases) * 0.15:
            issues.append({
                "issue": "Empty or very short outputs",
                "frequency": len(empty_failures),
                "severity": "HIGH",
                "suggestion": "Add minimum length requirement. Check for prompt clarity."
            })
        
        # Check for off-topic responses
        offtopic_failures = self.failure_patterns.get("off_topic", [])
        if len(offtopic_failures) > len(self.failure_cases) * 0.15:
            issues.append({
                "issue": "Off-topic or irrelevant responses",
                "frequency": len(offtopic_failures),
                "severity": "MEDIUM",
                "suggestion": "Add domain constraints. Provide more context in prompt."
            })
        
        # Check for timeouts
        timeout_failures = self.failure_patterns.get("timeout", [])
        if len(timeout_failures) > 0:
            issues.append({
                "issue": "Timeouts during generation",
                "frequency": len(timeout_failures),
                "severity": "MEDIUM",
                "suggestion": "Reduce prompt complexity or max_tokens. Simplify task."
            })
        
        return issues
    
    def document_failure_scenarios(self, num_examples: int = 5) -> str:
        """
        Document example failure scenarios.
        
        Args:
            num_examples: Number of examples per failure type
            
        Returns:
            Documentation text
        """
        doc = f"""
{'='*70}
FAILURE SCENARIO DOCUMENTATION
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Failures Documented: {len(self.failure_cases)}
{'='*70}

"""
        
        for failure_type, cases in self.failure_patterns.items():
            doc += f"""
{failure_type.upper().replace('_', ' ')}
{'-'*70}
Total Cases: {len(cases)}

Example Scenarios:
"""
            
            for i, case in enumerate(cases[:num_examples], 1):
                doc += f"""
Scenario {i}:
  Task: {case['task']}
  Input: {str(case['input'])[:100]}...
  Expected: {str(case['expected_output'])[:100]}...
  Output: {case['output'][:100]}...
  
  Root Cause: {self._diagnose_root_cause(case)}
  
  Suggested Fix: {self._suggest_fix(case)}

"""
        
        doc += f"{'='*70}\n"
        
        return doc
    
    def _diagnose_root_cause(self, failure_case: Dict) -> str:
        """Diagnose root cause of failure."""
        failure_type = failure_case["failure_type"]
        
        diagnoses = {
            "format_error": "LLM did not follow the requested output format. Likely due to unclear format instructions or lack of examples.",
            "empty_output": "LLM produced no meaningful output. Possible causes: prompt too complex, unclear instructions, or task beyond model capabilities.",
            "off_topic": "LLM generated content unrelated to the query. Likely due to insufficient context or domain specification.",
            "timeout": "Generation exceeded time limit. Prompt may be too complex or requesting too much content.",
            "incorrect_content": "LLM understood the format but generated factually incorrect or irrelevant content. May need better context or examples.",
            "partial_correctness": "LLM partially addressed the query but missed key elements. Instructions may need to be more explicit.",
            "refusal": "LLM refused to complete the task. Prompt may trigger safety mechanisms or be ambiguous.",
            "token_limit": "Output exceeded token limit. Need to request more concise output or increase limit.",
            "execution_error": "Technical error during execution. Check API connectivity and parameters."
        }
        
        return diagnoses.get(failure_type, "Unknown failure cause. Manual investigation needed.")
    
    def _suggest_fix(self, failure_case: Dict) -> str:
        """Suggest fix for the failure."""
        failure_type = failure_case["failure_type"]
        
        fixes = {
            "format_error": "Add explicit format template with example. Use phrases like 'Return ONLY valid JSON' or 'Format as:'",
            "empty_output": "Break down complex tasks into simpler steps. Add explicit instruction to provide detailed response.",
            "off_topic": "Add domain specification ('You are an environmental sustainability expert'). Provide relevant context.",
            "timeout": "Reduce requested output length. Simplify prompt structure. Increase timeout if possible.",
            "incorrect_content": "Add few-shot examples of correct outputs. Provide more domain-specific context.",
            "partial_correctness": "Make requirements more explicit with numbered steps. Use checklist format.",
            "refusal": "Rephrase prompt to be clearer about intent. Remove ambiguous phrasing.",
            "token_limit": "Request more concise output or increase max_tokens parameter.",
            "execution_error": "Check API parameters and connectivity. Retry with backoff strategy."
        }
        
        return fixes.get(failure_type, "Review prompt structure and model parameters.")
    
    def generate_improvement_recommendations(self) -> List[Dict]:
        """
        Generate prioritized recommendations for improvement.
        
        Returns:
            List of recommendations
        """
        analysis = self.analyze_patterns()
        recommendations = []
        
        # Based on failure distribution
        failure_dist = analysis.get("failure_distribution", {})
        
        for failure_type, count in sorted(failure_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / analysis["total_failures"]) * 100
            
            if percentage > 20:
                priority = "HIGH"
            elif percentage > 10:
                priority = "MEDIUM"
            else:
                priority = "LOW"
            
            recommendations.append({
                "priority": priority,
                "failure_type": failure_type,
                "frequency": count,
                "percentage": f"{percentage:.1f}%",
                "action": self._get_improvement_action(failure_type)
            })
        
        # Sort by priority
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        recommendations.sort(key=lambda x: priority_order[x["priority"]])
        
        return recommendations
    
    def _get_improvement_action(self, failure_type: str) -> str:
        """Get improvement action for failure type."""
        actions = {
            "format_error": "Implement stricter format templates and post-processing validation",
            "empty_output": "Add output length verification and retry logic",
            "off_topic": "Enhance context provision and add relevance checks",
            "timeout": "Implement prompt optimization and chunking strategies",
            "incorrect_content": "Expand few-shot examples and domain-specific training data",
            "partial_correctness": "Add requirement checklists and validation steps",
            "refusal": "Refine prompt phrasing and add clarifying context",
            "token_limit": "Implement dynamic token allocation based on task",
            "execution_error": "Add robust error handling and retry mechanisms"
        }
        
        return actions.get(failure_type, "Investigate and document specific failure pattern")
    
    def export_failure_analysis(self, filename: Optional[str] = None) -> str:
        """Export failure analysis."""
        if not filename:
            filename = f"failure_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.results_dir / filename
        
        analysis = self.analyze_patterns()
        recommendations = self.generate_improvement_recommendations()
        
        export_data = {
            "analysis": analysis,
            "recommendations": recommendations,
            "failure_cases": self.failure_cases,
            "patterns": {k: len(v) for k, v in self.failure_patterns.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f" Failure analysis exported to {filepath}")
        return str(filepath)
    
    def generate_report(self) -> str:
        """Generate comprehensive failure analysis report."""
        analysis = self.analyze_patterns()
        recommendations = self.generate_improvement_recommendations()
        
        report = f"""
{'='*70}
FAILURE ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

OVERVIEW:
- Total Failures: {analysis.get('total_failures', 0)}
- Failure Types: {len(analysis.get('failure_distribution', {}))}

FAILURE DISTRIBUTION:
"""
        
        for failure_type, count in analysis.get('failure_distribution', {}).items():
            percentage = (count / analysis['total_failures']) * 100
            report += f"  - {failure_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)\n"
        
        report += f"""
TASK-SPECIFIC FAILURES:
"""
        
        for task, count in analysis.get('task_failures', {}).items():
            report += f"  - {task}: {count} failures\n"
        
        report += f"""
COMMON ISSUES IDENTIFIED:
"""
        
        for issue in analysis.get('common_issues', []):
            report += f"""
  [{issue['severity']}] {issue['issue']}
    Frequency: {issue['frequency']} occurrences
    Suggestion: {issue['suggestion']}
"""
        
        report += f"""
PRIORITIZED RECOMMENDATIONS:
"""
        
        for rec in recommendations:
            report += f"""
  [{rec['priority']}] {rec['failure_type'].replace('_', ' ').title()}
    Frequency: {rec['frequency']} ({rec['percentage']})
    Action: {rec['action']}
"""
        
        report += f"\n{'='*70}\n"
        
        return report
