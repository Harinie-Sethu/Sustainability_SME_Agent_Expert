"""
Observations Logger for Recording Agent Performance
Tracks iterations, decisions, and refinements for prompt improvement
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ObservationsLogger:
    """
    Logs observations for:
    - Agent decision-making process
    - Task execution results
    - Error patterns
    - Performance metrics
    - Prompt effectiveness
    """
    
    def __init__(self, log_dir: str = "parte/observations_log"):
        """
        Initialize observations logger.
        
        Args:
            log_dir: Directory for observation logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.observations: List[Dict] = []
        self.iteration_count = 0
        
        logger.info(f" Observations logger initialized: {self.log_dir}")
    
    def log_planning(self, user_request: str, plan: Dict[str, Any], 
                    context: Optional[str] = None):
        """
        Log planning phase.
        
        Args:
            user_request: User's request
            plan: Generated plan
            context: Conversation context
        """
        observation = {
            "type": "planning",
            "timestamp": datetime.now().isoformat(),
            "iteration": self.iteration_count,
            "user_request": user_request,
            "plan": plan,
            "context_used": context is not None,
            "num_subtasks": len(plan.get('subtasks', [])),
            "capabilities_required": list(set(
                st.get('capability') for st in plan.get('subtasks', [])
            ))
        }
        
        self.observations.append(observation)
        logger.info(f" Logged planning observation (iteration {self.iteration_count})")
    
    def log_routing(self, user_request: str, routing_decision: Dict[str, Any],
                   rule_based: Optional[Dict] = None, llm_based: Optional[Dict] = None):
        """
        Log routing decision.
        
        Args:
            user_request: User's request
            routing_decision: Final routing decision
            rule_based: Rule-based routing result
            llm_based: LLM-based routing result
        """
        observation = {
            "type": "routing",
            "timestamp": datetime.now().isoformat(),
            "iteration": self.iteration_count,
            "user_request": user_request,
            "final_handler": routing_decision.get('handler'),
            "confidence": routing_decision.get('confidence'),
            "reasoning": routing_decision.get('reasoning'),
            "rule_based_handler": rule_based.get('handler') if rule_based else None,
            "llm_based_handler": llm_based.get('handler') if llm_based else None,
            "agreement": (
                rule_based.get('handler') == llm_based.get('handler')
                if rule_based and llm_based else None
            )
        }
        
        self.observations.append(observation)
        logger.info(f" Logged routing observation")
    
    def log_execution(self, handler_name: str, parameters: Dict[str, Any],
                     result: Dict[str, Any], execution_time: float):
        """
        Log task execution.
        
        Args:
            handler_name: Handler that was executed
            parameters: Parameters passed to handler
            result: Execution result
            execution_time: Time taken in seconds
        """
        observation = {
            "type": "execution",
            "timestamp": datetime.now().isoformat(),
            "iteration": self.iteration_count,
            "handler": handler_name,
            "parameters": parameters,
            "success": result.get('success', False),
            "execution_time": execution_time,
            "error": result.get('error') if not result.get('success') else None
        }
        
        self.observations.append(observation)
        logger.info(f" Logged execution observation")
    
    def log_error(self, error_type: str, error_message: str, context: Dict[str, Any]):
        """
        Log error occurrence.
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Context when error occurred
        """
        observation = {
            "type": "error",
            "timestamp": datetime.now().isoformat(),
            "iteration": self.iteration_count,
            "error_type": error_type,
            "error_message": error_message,
            "context": context
        }
        
        self.observations.append(observation)
        logger.warning(f"️ Logged error observation: {error_type}")
    
    def log_reasoning(self, problem: str, reasoning_steps: List[str],
                     conclusion: str, confidence: str):
        """
        Log reasoning process.
        
        Args:
            problem: Problem being reasoned about
            reasoning_steps: Steps in reasoning process
            conclusion: Final conclusion
            confidence: Confidence level
        """
        observation = {
            "type": "reasoning",
            "timestamp": datetime.now().isoformat(),
            "iteration": self.iteration_count,
            "problem": problem,
            "num_steps": len(reasoning_steps),
            "reasoning_steps": reasoning_steps,
            "conclusion": conclusion,
            "confidence": confidence
        }
        
        self.observations.append(observation)
        logger.info(f" Logged reasoning observation")
    
    def log_reflection(self, user_request: str, agent_response: str,
                      reflection_analysis: Dict[str, Any]):
        """
        Log reflection on agent performance.
        
        Args:
            user_request: Original user request
            agent_response: Agent's response
            reflection_analysis: Analysis from reflection
        """
        observation = {
            "type": "reflection",
            "timestamp": datetime.now().isoformat(),
            "iteration": self.iteration_count,
            "user_request": user_request,
            "agent_response": agent_response[:200],  # Preview
            "analysis": reflection_analysis
        }
        
        self.observations.append(observation)
        logger.info(f" Logged reflection observation")
    
    def increment_iteration(self):
        """Increment iteration counter for new interaction."""
        self.iteration_count += 1
    
    def save_observations(self, filename: Optional[str] = None):
        """
        Save observations to file.
        
        Args:
            filename: Optional filename
        """
        if not filename:
            filename = f"observations_{self.current_session}.json"
        
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump({
                "session": self.current_session,
                "total_iterations": self.iteration_count,
                "observations": self.observations
            }, f, indent=2)
        
        logger.info(f"💾 Saved {len(self.observations)} observations to {filepath}")
        return str(filepath)
    
    def generate_insights(self) -> Dict[str, Any]:
        """
        Generate insights from logged observations.
        
        Returns:
            Dict with insights and metrics
        """
        if not self.observations:
            return {"message": "No observations to analyze"}
        
        insights = {
            "total_observations": len(self.observations),
            "total_iterations": self.iteration_count,
            "observation_types": {},
            "routing_accuracy": {},
            "execution_success_rate": 0,
            "common_errors": [],
            "handler_usage": {}
        }
        
        # Count observation types
        for obs in self.observations:
            obs_type = obs.get('type', 'unknown')
            insights['observation_types'][obs_type] = insights['observation_types'].get(obs_type, 0) + 1
        
        # Analyze routing
        routing_obs = [o for o in self.observations if o['type'] == 'routing']
        if routing_obs:
            agreements = sum(1 for o in routing_obs if o.get('agreement') == True)
            total_with_comparison = sum(1 for o in routing_obs if o.get('agreement') is not None)
            if total_with_comparison > 0:
                insights['routing_accuracy']['agreement_rate'] = agreements / total_with_comparison
            
            confidence_dist = {}
            for o in routing_obs:
                conf = o.get('confidence', 'unknown')
                confidence_dist[conf] = confidence_dist.get(conf, 0) + 1
            insights['routing_accuracy']['confidence_distribution'] = confidence_dist
        
        # Analyze execution
        execution_obs = [o for o in self.observations if o['type'] == 'execution']
        if execution_obs:
            successes = sum(1 for o in execution_obs if o.get('success', False))
            insights['execution_success_rate'] = successes / len(execution_obs)
            
            for o in execution_obs:
                handler = o.get('handler', 'unknown')
                insights['handler_usage'][handler] = insights['handler_usage'].get(handler, 0) + 1
        
        # Analyze errors
        error_obs = [o for o in self.observations if o['type'] == 'error']
        error_types = {}
        for o in error_obs:
            error_type = o.get('error_type', 'unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        insights['common_errors'] = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return insights
    
    def generate_report(self, save_to_file: bool = True) -> str:
        """
        Generate human-readable report.
        
        Args:
            save_to_file: Whether to save report to file
            
        Returns:
            Report text
        """
        insights = self.generate_insights()
        
        report = f"""
{'='*70}
AGENT OBSERVATIONS REPORT
Session: {self.current_session}
{'='*70}

SUMMARY:
- Total Iterations: {insights.get('total_iterations', 0)}
- Total Observations: {insights.get('total_observations', 0)}

OBSERVATION BREAKDOWN:
"""
        
        for obs_type, count in insights.get('observation_types', {}).items():
            report += f"  - {obs_type.title()}: {count}\n"
        
        report += f"""
ROUTING PERFORMANCE:
"""
        routing_acc = insights.get('routing_accuracy', {})
        if routing_acc:
            agreement_rate = routing_acc.get('agreement_rate', 0)
            report += f"  - Rule-based vs LLM Agreement: {agreement_rate:.1%}\n"
            
            conf_dist = routing_acc.get('confidence_distribution', {})
            report += f"  - Confidence Distribution:\n"
            for conf, count in conf_dist.items():
                report += f"    - {conf}: {count}\n"
        else:
            report += "  - No routing data available\n"
        
        report += f"""
EXECUTION PERFORMANCE:
  - Success Rate: {insights.get('execution_success_rate', 0):.1%}
  - Handler Usage:
"""
        for handler, count in insights.get('handler_usage', {}).items():
            report += f"    - {handler}: {count}\n"
        
        report += f"""
ERROR ANALYSIS:
"""
        common_errors = insights.get('common_errors', [])
        if common_errors:
            for error_type, count in common_errors:
                report += f"  - {error_type}: {count} occurrences\n"
        else:
            report += "  - No errors logged\n"
        
        report += f"""
{'='*70}
"""
        
        if save_to_file:
            report_file = self.log_dir / f"report_{self.current_session}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            logger.info(f" Report saved to {report_file}")
        
        return report
    
    def get_prompt_refinement_suggestions(self) -> List[str]:
        """
        Analyze observations to suggest prompt refinements.
        
        Returns:
            List of suggestions for improving prompts
        """
        suggestions = []
        
        # Analyze routing disagreements
        routing_obs = [o for o in self.observations if o['type'] == 'routing']
        disagreements = [o for o in routing_obs if o.get('agreement') == False]
        
        if len(disagreements) > len(routing_obs) * 0.3:  # More than 30% disagreement
            suggestions.append(
                "HIGH: Routing prompt needs refinement - high disagreement between "
                "rule-based and LLM routing. Consider adding more explicit examples."
            )
        
        # Analyze low confidence routes
        low_confidence = [o for o in routing_obs if o.get('confidence') == 'low']
        if len(low_confidence) > len(routing_obs) * 0.2:  # More than 20% low confidence
            suggestions.append(
                "MEDIUM: Consider adding more specific routing patterns or examples "
                "to improve confidence in routing decisions."
            )
        
        # Analyze execution failures
        execution_obs = [o for o in self.observations if o['type'] == 'execution']
        failures = [o for o in execution_obs if not o.get('success', False)]
        
        if len(failures) > len(execution_obs) * 0.2:  # More than 20% failure
            suggestions.append(
                "HIGH: Execution failure rate is high. Review task-specific prompts "
                "and add more detailed instructions or error handling."
            )
        
        # Analyze reasoning quality
        reasoning_obs = [o for o in self.observations if o['type'] == 'reasoning']
        low_conf_reasoning = [o for o in reasoning_obs if o.get('confidence') == 'low']
        
        if len(low_conf_reasoning) > len(reasoning_obs) * 0.3:
            suggestions.append(
                "MEDIUM: Reasoning confidence is low. Consider adding more structured "
                "reasoning templates or chain-of-thought examples."
            )
        
        # Analyze common errors
        error_obs = [o for o in self.observations if o['type'] == 'error']
        if len(error_obs) > 0:
            error_types = {}
            for o in error_obs:
                error_type = o.get('error_type', 'unknown')
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            most_common_error = max(error_types.items(), key=lambda x: x[1])
            suggestions.append(
                f"HIGH: Most common error is '{most_common_error[0]}' "
                f"({most_common_error[1]} occurrences). Add specific handling for this case."
            )
        
        if not suggestions:
            suggestions.append(" No major issues detected. Prompts are performing well.")
        
        return suggestions
