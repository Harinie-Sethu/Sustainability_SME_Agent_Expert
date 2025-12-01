"""
Experiment Runner for Systematic Evaluation
Orchestrates all experiments and generates comprehensive reports
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime

from partf.prompt_library import PromptLibrary
from partf.few_shot_examples import FewShotLibrary
from partf.prompt_engineer import PromptEngineer
from partf.model_evaluator import ModelEvaluator
from partf.ablation_studies import AblationStudies
from partf.failure_analyzer import FailureAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Orchestrate comprehensive experiments:
    - Prompt variant comparison
    - Few-shot vs zero-shot
    - Temperature/token experiments
    - Ablation studies
    - Failure analysis
    """
    
    def __init__(self, llm_client, rag_system=None):
        """
        Initialize experiment runner.
        
        Args:
            llm_client: LLM client
            rag_system: Optional RAG system for context
        """
        self.llm = llm_client
        self.rag = rag_system
        
        # Initialize components
        self.prompt_lib = PromptLibrary()
        self.few_shot_lib = FewShotLibrary()
        self.prompt_engineer = PromptEngineer(llm_client, self.prompt_lib, self.few_shot_lib)
        self.model_evaluator = ModelEvaluator()
        self.ablation_studies = AblationStudies(llm_client)
        self.failure_analyzer = FailureAnalyzer()
        
        # Results storage
        self.experiment_results: Dict[str, Any] = {
            "metadata": {
                "started": datetime.now().isoformat(),
                "llm_model": getattr(llm_client, 'model_name', 'unknown')
            },
            "experiments": {}
        }
        
        logger.info("✓ Experiment Runner initialized")
    
    def run_prompt_variant_experiments(self, task: str, test_cases: List[Dict]) -> Dict[str, Any]:
        """
        Run experiments comparing prompt variants.
        
        Args:
            task: Task type (qa, quiz_generation, etc.)
            test_cases: Test cases
            
        Returns:
            Experiment results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"EXPERIMENT: Prompt Variant Comparison - {task}")
        logger.info(f"{'='*70}")
        
        # Get available variants for task
        variants = list(self.prompt_lib.prompts.get(task, {}).keys())
        
        if not variants:
            logger.warning(f"No variants available for task: {task}")
            return {"error": "No variants available"}
        
        logger.info(f"Testing {len(variants)} variants: {variants}")
        
        # Compare variants
        results = self.prompt_engineer.compare_variants(task, variants, test_cases)
        
        # Store results
        self.experiment_results["experiments"][f"prompt_variants_{task}"] = results
        
        # Log failures
        for variant, variant_results in results["variant_results"].items():
            for result in variant_results:
                if not result.get('success'):
                    test_case = test_cases[variant_results.index(result)]
                    self.failure_analyzer.record_failure(
                        task=task,
                        prompt=result.get('prompt', ''),
                        input_data=test_case.get('input', test_case),
                        output=result.get('output', ''),
                        expected_output=test_case.get('expected_output'),
                        error=result.get('error')
                    )
        
        logger.info(f"✓ Best variant: {results['best_variant']['name']}")
        logger.info(f"  Success rate: {results['best_variant']['metrics']['success_rate']:.1%}")
        
        return results
    
    def run_few_shot_experiments(self, task: str, test_cases: List[Dict],
                                 shot_counts: List[int] = None) -> Dict[str, Any]:
        """
        Run few-shot learning experiments.
        
        Args:
            task: Task type
            test_cases: Test cases
            shot_counts: Number of shots to test (e.g., [0, 1, 3, 5])
            
        Returns:
            Experiment results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"EXPERIMENT: Few-Shot Learning - {task}")
        logger.info(f"{'='*70}")
        
        if shot_counts is None:
            shot_counts = [0, 1, 3, 5]
        
        results = {
            "task": task,
            "shot_counts": shot_counts,
            "results": {}
        }
        
        for num_shots in shot_counts:
            logger.info(f"\nTesting {num_shots}-shot learning")
            
            shot_results = []
            for test_case in test_cases:
                result = self.prompt_engineer.test_few_shot(
                    task=task,
                    num_shots=num_shots,
                    test_input=test_case.get('input', test_case),
                    expected_output=test_case.get('expected_output')
                )
                shot_results.append(result)
            
            # Calculate metrics
            successful = [r for r in shot_results if r.get('success')]
            results["results"][str(num_shots)] = {
                "success_rate": len(successful) / len(shot_results),
                "avg_time": sum(r.get('execution_time', 0) for r in successful) / len(successful) if successful else 0,
                "results": shot_results
            }
            
            logger.info(f"  Success rate: {results['results'][str(num_shots)]['success_rate']:.1%}")
        
        # Determine optimal shot count
        best_shot = max(results["results"].items(), key=lambda x: x[1]["success_rate"])
        results["optimal_shot_count"] = {
            "count": int(best_shot[0]),
            "success_rate": best_shot[1]["success_rate"]
        }
        
        self.experiment_results["experiments"][f"few_shot_{task}"] = results
        
        logger.info(f"✓ Optimal shot count: {results['optimal_shot_count']['count']}")
        
        return results
    
    def run_temperature_experiments(self, task: str, test_cases: List[Dict],
                                   temperatures: List[float] = None) -> Dict[str, Any]:
        """
        Run temperature comparison experiments.
        
        Args:
            task: Task type
            test_cases: Test cases
            temperatures: Temperatures to test
            
        Returns:
            Experiment results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"EXPERIMENT: Temperature Comparison - {task}")
        logger.info(f"{'='*70}")
        
        if temperatures is None:
            temperatures = [0.3, 0.5, 0.7, 0.9]
        
        results = self.model_evaluator.compare_temperatures(
            self.llm, task, test_cases, temperatures
        )
        
        self.experiment_results["experiments"][f"temperature_{task}"] = results
        
        logger.info(f"✓ Best temperature: {results['best_temperature']['value']}")
        
        return results
    
    def run_ablation_experiments(self, task: str, full_prompt: str,
                                test_cases: List[Dict]) -> Dict[str, Any]:
        """
        Run ablation study experiments.
        
        Args:
            task: Task type
            full_prompt: Full featured prompt
            test_cases: Test cases
            
        Returns:
            Experiment results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"EXPERIMENT: Ablation Studies - {task}")
        logger.info(f"{'='*70}")
        
        results = self.ablation_studies.comprehensive_ablation(
            full_prompt, test_cases
        )
        
        self.experiment_results["experiments"][f"ablation_{task}"] = results
        
        # Log component importance
        logger.info("\nComponent Importance Ranking:")
        for i, comp in enumerate(results['component_importance'], 1):
            logger.info(f"  {i}. {comp['component']}: {comp['importance_score']:.1%}")
        
        return results
    
    def run_comprehensive_experiments(self, tasks: List[str],
                                     test_cases_per_task: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Run comprehensive experiments across multiple tasks.
        
        Args:
            tasks: List of task names
            test_cases_per_task: Test cases for each task
            
        Returns:
            All experiment results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"RUNNING COMPREHENSIVE EXPERIMENTS")
        logger.info(f"Tasks: {', '.join(tasks)}")
        logger.info(f"{'='*70}")
        
        for task in tasks:
            test_cases = test_cases_per_task.get(task, [])
            
            if not test_cases:
                logger.warning(f"No test cases for task: {task}")
                continue
            
            logger.info(f"\n\n{'#'*70}")
            logger.info(f"TASK: {task.upper()}")
            logger.info(f"{'#'*70}")
            
            # 1. Prompt variant comparison
            try:
                self.run_prompt_variant_experiments(task, test_cases)
            except Exception as e:
                logger.error(f"Prompt variant experiments failed: {e}")
            
            # 2. Few-shot experiments
            try:
                self.run_few_shot_experiments(task, test_cases)
            except Exception as e:
                logger.error(f"Few-shot experiments failed: {e}")
            
            # 3. Temperature experiments
            try:
                self.run_temperature_experiments(task, test_cases)
            except Exception as e:
                logger.error(f"Temperature experiments failed: {e}")
        
        # Finalize results
        self.experiment_results["metadata"]["completed"] = datetime.now().isoformat()
        
        return self.experiment_results
    
    def generate_comprehensive_report(self) -> str:
        """
        Generate comprehensive experiment report.
        
        Returns:
            Report text
        """
        report = f"""
{'='*80}
COMPREHENSIVE EXPERIMENT REPORT
LLM Model: {self.experiment_results['metadata']['llm_model']}
Started: {self.experiment_results['metadata']['started']}
Completed: {self.experiment_results['metadata'].get('completed', 'In Progress')}
{'='*80}

"""
        
        # Experiments summary
        experiments = self.experiment_results.get('experiments', {})
        report += f"Total Experiments: {len(experiments)}\n\n"
        
        # Prompt variant results
        report += f"""
{'='*80}
1. PROMPT VARIANT EXPERIMENTS
{'='*80}
"""
        
        for exp_name, exp_data in experiments.items():
            if 'prompt_variants' in exp_name:
                task = exp_data.get('task', 'unknown')
                best = exp_data.get('best_variant', {})
                
                report += f"""
Task: {task}
Best Variant: {best.get('name', 'unknown')}
  - Success Rate: {best.get('metrics', {}).get('success_rate', 0):.1%}
  - Avg Time: {best.get('metrics', {}).get('avg_execution_time', 0):.3f}s

All Variants Performance:
"""
                
                for variant, metrics in exp_data.get('summary', {}).items():
                    report += f"  {variant}: {metrics.get('success_rate', 0):.1%} success\n"
        
        # Few-shot results
        report += f"""
{'='*80}
2. FEW-SHOT LEARNING EXPERIMENTS
{'='*80}
"""
        
        for exp_name, exp_data in experiments.items():
            if 'few_shot' in exp_name:
                task = exp_data.get('task', 'unknown')
                optimal = exp_data.get('optimal_shot_count', {})
                
                report += f"""
Task: {task}
Optimal Shot Count: {optimal.get('count', 0)} shots
  - Success Rate: {optimal.get('success_rate', 0):.1%}

Shot Count Performance:
"""
                
                for shots, metrics in exp_data.get('results', {}).items():
                    report += f"  {shots}-shot: {metrics.get('success_rate', 0):.1%}\n"
        
        # Temperature results
        report += f"""
{'='*80}
3. TEMPERATURE EXPERIMENTS
{'='*80}
"""
        
        for exp_name, exp_data in experiments.items():
            if 'temperature' in exp_name:
                task = exp_data.get('task', 'unknown')
                best_temp = exp_data.get('best_temperature', {})
                
                report += f"""
Task: {task}
Best Temperature: {best_temp.get('value', 0.7)}
  - Success Rate: {best_temp.get('success_rate', 0):.1%}

Temperature Performance:
"""
                
                for temp, eval_data in exp_data.get('evaluations', {}).items():
                    report += f"  {temp}: {eval_data.get('success_rate', 0):.1%}\n"
        
        # Ablation results
        report += f"""
{'='*80}
4. ABLATION STUDIES
{'='*80}
"""
        
        for exp_name, exp_data in experiments.items():
            if 'ablation' in exp_name:
                report += f"\nComponent Importance:\n"
                for comp in exp_data.get('component_importance', []):
                    report += f"  {comp['component']}: {comp['importance_score']:.1%} impact\n"
        
        # Failure analysis
        report += f"""
{'='*80}
5. FAILURE ANALYSIS
{'='*80}
"""
        
        failure_report = self.failure_analyzer.generate_report()
        report += failure_report
        
        # Recommendations
        recommendations = self.failure_analyzer.generate_improvement_recommendations()
        report += f"""
{'='*80}
6. IMPROVEMENT RECOMMENDATIONS
{'='*80}
"""
        
        for rec in recommendations[:5]:  # Top 5 recommendations
            report += f"""
[{rec['priority']}] {rec['failure_type'].replace('_', ' ').title()}
  Frequency: {rec['frequency']} ({rec['percentage']})
  Action: {rec['action']}
"""
        
        report += f"\n{'='*80}\n"
        
        return report
    
    def export_all_results(self, base_filename: str = "comprehensive_experiments") -> Dict[str, str]:
        """
        Export all experiment results.
        
        Args:
            base_filename: Base filename for exports
            
        Returns:
            Dict of exported file paths
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exported_files = {}
        
        # Export main results
        results_file = f"partf/experiments/results/{base_filename}_{timestamp}.json"
        Path(results_file).parent.mkdir(exist_ok=True, parents=True)
        with open(results_file, 'w') as f:
            json.dump(self.experiment_results, f, indent=2)
        exported_files['results'] = results_file
        
        # Export report
        report = self.generate_comprehensive_report()
        report_file = f"partf/reports/{base_filename}_report_{timestamp}.txt"
        Path(report_file).parent.mkdir(exist_ok=True, parents=True)
        with open(report_file, 'w') as f:
            f.write(report)
        exported_files['report'] = report_file
        
        # Export failure analysis
        failure_file = self.failure_analyzer.export_failure_analysis(
            f"failure_analysis_{timestamp}.json"
        )
        exported_files['failures'] = failure_file
        
        # Export failure scenarios documentation
        scenarios = self.failure_analyzer.document_failure_scenarios()
        scenarios_file = f"partf/reports/failure_scenarios_{timestamp}.txt"
        with open(scenarios_file, 'w') as f:
            f.write(scenarios)
        exported_files['scenarios'] = scenarios_file
        
        # Export ablation studies
        if self.ablation_studies.studies:
            ablation_file = self.ablation_studies.export_studies(
                f"ablation_studies_{timestamp}.json"
            )
            exported_files['ablation'] = ablation_file
        
        logger.info(f"\n{'='*70}")
        logger.info("EXPORTED FILES:")
        for file_type, filepath in exported_files.items():
            logger.info(f"  {file_type}: {filepath}")
        logger.info(f"{'='*70}")
        
        return exported_files
