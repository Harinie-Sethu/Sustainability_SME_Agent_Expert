"""
Retrieval Evaluation Metrics
Evaluate quality of retrieval systems
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Dict, Any, Optional, Set
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """
    Evaluate retrieval quality with metrics:
    - Precision@K
    - Recall@K
    - Mean Reciprocal Rank (MRR)
    - Normalized Discounted Cumulative Gain (NDCG)
    - Mean Average Precision (MAP)
    """
    
    def __init__(self):
        """Initialize retrieval evaluator."""
        logger.info(" Retrieval Evaluator initialized")
    
    def precision_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Calculate Precision@K.
        
        Args:
            retrieved: List of retrieved document IDs (in order)
            relevant: Set of relevant document IDs
            k: K value
            
        Returns:
            Precision@K score
        """
        if k == 0 or not retrieved:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
        
        return relevant_retrieved / k
    
    def recall_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: K value
            
        Returns:
            Recall@K score
        """
        if not relevant or k == 0:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
        
        return relevant_retrieved / len(relevant)
    
    def f1_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Calculate F1@K.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: K value
            
        Returns:
            F1@K score
        """
        precision = self.precision_at_k(retrieved, relevant, k)
        recall = self.recall_at_k(retrieved, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def mean_reciprocal_rank(self, retrieved: List[str], relevant: Set[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            
        Returns:
            MRR score
        """
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / rank
        
        return 0.0
    
    def average_precision(self, retrieved: List[str], relevant: Set[str]) -> float:
        """
        Calculate Average Precision (AP).
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            
        Returns:
            AP score
        """
        if not relevant:
            return 0.0
        
        score = 0.0
        num_relevant_seen = 0
        
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                num_relevant_seen += 1
                precision_at_rank = num_relevant_seen / rank
                score += precision_at_rank
        
        return score / len(relevant)
    
    def ndcg_at_k(self, retrieved: List[str], relevant: Dict[str, float], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Dict mapping document IDs to relevance scores (0-1 or graded)
            k: K value
            
        Returns:
            NDCG@K score
        """
        if k == 0 or not retrieved or not relevant:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved[:k], start=1):
            relevance = relevant.get(doc_id, 0.0)
            dcg += relevance / np.log2(rank + 1)
        
        # Calculate IDCG (ideal DCG)
        ideal_relevances = sorted(relevant.values(), reverse=True)[:k]
        idcg = 0.0
        for rank, relevance in enumerate(ideal_relevances, start=1):
            idcg += relevance / np.log2(rank + 1)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate_retrieval(self, retrieved: List[str], relevant: Set[str],
                          k_values: List[int] = None) -> Dict[str, Any]:
        """
        Comprehensive retrieval evaluation.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k_values: K values to evaluate (default: [1, 5, 10])
            
        Returns:
            Evaluation metrics
        """
        if k_values is None:
            k_values = [1, 5, 10]
        
        results = {
            'precision_at_k': {},
            'recall_at_k': {},
            'f1_at_k': {},
            'mrr': self.mean_reciprocal_rank(retrieved, relevant),
            'map': self.average_precision(retrieved, relevant)
        }
        
        for k in k_values:
            results['precision_at_k'][k] = self.precision_at_k(retrieved, relevant, k)
            results['recall_at_k'][k] = self.recall_at_k(retrieved, relevant, k)
            results['f1_at_k'][k] = self.f1_at_k(retrieved, relevant, k)
        
        return results
    
    def compare_retrievers(self, retriever_results: Dict[str, List[str]],
                          relevant: Set[str],
                          k_values: List[int] = None) -> Dict[str, Any]:
        """
        Compare multiple retrievers.
        
        Args:
            retriever_results: Dict mapping retriever names to retrieved doc lists
            relevant: Set of relevant document IDs
            k_values: K values to evaluate
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(retriever_results)} retrievers")
        
        comparison = {
            'retrievers': {},
            'best_by_metric': {}
        }
        
        # Evaluate each retriever
        for retriever_name, retrieved in retriever_results.items():
            metrics = self.evaluate_retrieval(retrieved, relevant, k_values)
            comparison['retrievers'][retriever_name] = metrics
        
        # Determine best for each metric
        if k_values is None:
            k_values = [1, 5, 10]
        
        for k in k_values:
            best_precision = max(comparison['retrievers'].items(),
                               key=lambda x: x[1]['precision_at_k'][k])
            comparison['best_by_metric'][f'precision@{k}'] = best_precision[0]
            
            best_recall = max(comparison['retrievers'].items(),
                            key=lambda x: x[1]['recall_at_k'][k])
            comparison['best_by_metric'][f'recall@{k}'] = best_recall[0]
        
        best_mrr = max(comparison['retrievers'].items(),
                      key=lambda x: x[1]['mrr'])
        comparison['best_by_metric']['mrr'] = best_mrr[0]
        
        best_map = max(comparison['retrievers'].items(),
                      key=lambda x: x[1]['map'])
        comparison['best_by_metric']['map'] = best_map[0]
        
        return comparison
    
    def calculate_diversity(self, retrieved_docs: List[Dict]) -> float:
        """
        Calculate diversity of retrieved documents.
        
        Args:
            retrieved_docs: List of retrieved documents with text
            
        Returns:
            Diversity score (0-1)
        """
        if len(retrieved_docs) < 2:
            return 0.0
        
        # Calculate pairwise similarities
        similarities = []
        
        for i in range(len(retrieved_docs)):
            for j in range(i + 1, len(retrieved_docs)):
                text_i = set(retrieved_docs[i].get('text', '').lower().split())
                text_j = set(retrieved_docs[j].get('text', '').lower().split())
                
                if text_i and text_j:
                    # Jaccard similarity
                    intersection = len(text_i & text_j)
                    union = len(text_i | text_j)
                    similarity = intersection / union if union > 0 else 0
                    similarities.append(similarity)
        
        if not similarities:
            return 0.0
        
        # Diversity is inverse of average similarity
        avg_similarity = np.mean(similarities)
        diversity = 1.0 - avg_similarity
        
        return diversity
    
    def generate_report(self, evaluation_results: Dict[str, Any],
                       retriever_name: str = "Retriever") -> str:
        """
        Generate human-readable evaluation report.
        
        Args:
            evaluation_results: Evaluation results
            retriever_name: Name of the retriever
            
        Returns:
            Report text
        """
        report = f"""
{'='*70}
RETRIEVAL EVALUATION REPORT: {retriever_name}
{'='*70}

PRECISION@K:
"""
        
        for k, score in evaluation_results['precision_at_k'].items():
            report += f"  P@{k}: {score:.3f}\n"
        
        report += "\nRECALL@K:\n"
        for k, score in evaluation_results['recall_at_k'].items():
            report += f"  R@{k}: {score:.3f}\n"
        
        report += "\nF1@K:\n"
        for k, score in evaluation_results['f1_at_k'].items():
            report += f"  F1@{k}: {score:.3f}\n"
        
        report += f"\nMEAN RECIPROCAL RANK (MRR): {evaluation_results['mrr']:.3f}\n"
        report += f"MEAN AVERAGE PRECISION (MAP): {evaluation_results['map']:.3f}\n"
        
        report += f"\n{'='*70}\n"
        
        return report
