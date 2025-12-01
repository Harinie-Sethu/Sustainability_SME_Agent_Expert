"""
Ranking Fusion Strategies
Advanced methods for combining multiple ranking signals
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Dict, Any, Optional
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RankingFusion:
    """
    Advanced ranking fusion strategies:
    - Reciprocal Rank Fusion (RRF)
    - Weighted combination
    - Borda count
    - Score normalization and combination
    """
    
    def __init__(self):
        """Initialize ranking fusion."""
        logger.info(" Ranking Fusion initialized")
    
    def reciprocal_rank_fusion(self, ranked_lists: List[List[Dict]], 
                               k: int = 60) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF).
        
        RRF formula: score = sum(1 / (k + rank_i)) for all lists
        
        Args:
            ranked_lists: List of ranked result lists
            k: Constant for RRF (typically 60)
            
        Returns:
            Fused ranked list
        """
        logger.info(f"Applying Reciprocal Rank Fusion (k={k})")
        
        # Build score map
        score_map = {}
        
        for ranked_list in ranked_lists:
            for rank, result in enumerate(ranked_list, start=1):
                key = self._get_key(result)
                
                # RRF score
                rrf_score = 1.0 / (k + rank)
                
                if key not in score_map:
                    score_map[key] = {
                        'result': result,
                        'rrf_score': 0.0,
                        'appearances': 0
                    }
                
                score_map[key]['rrf_score'] += rrf_score
                score_map[key]['appearances'] += 1
        
        # Create fused list
        fused_results = []
        for key, data in score_map.items():
            result = data['result'].copy()
            result['rrf_score'] = data['rrf_score']
            result['list_appearances'] = data['appearances']
            fused_results.append(result)
        
        # Sort by RRF score
        fused_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        
        logger.info(f"  Fused {len(fused_results)} unique results")
        
        return fused_results
    
    def weighted_fusion(self, ranked_lists: List[List[Dict]],
                       weights: List[float],
                       score_fields: List[str]) -> List[Dict[str, Any]]:
        """
        Weighted score fusion.
        
        Args:
            ranked_lists: List of ranked result lists
            weights: Weights for each list
            score_fields: Score field names for each list
            
        Returns:
            Fused ranked list
        """
        logger.info(f"Applying Weighted Fusion (weights={weights})")
        
        if len(ranked_lists) != len(weights) or len(ranked_lists) != len(score_fields):
            raise ValueError("ranked_lists, weights, and score_fields must have same length")
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Build score map
        score_map = {}
        
        for i, (ranked_list, weight, score_field) in enumerate(zip(ranked_lists, normalized_weights, score_fields)):
            # Normalize scores for this list
            scores = [r.get(score_field, 0) for r in ranked_list]
            if not scores:
                continue
            
            max_score = max(scores)
            min_score = min(scores)
            score_range = max_score - min_score if max_score > min_score else 1.0
            
            for result in ranked_list:
                key = self._get_key(result)
                score = result.get(score_field, 0)
                normalized_score = (score - min_score) / score_range
                weighted_score = normalized_score * weight
                
                if key not in score_map:
                    score_map[key] = {
                        'result': result,
                        'weighted_score': 0.0,
                        'component_scores': {}
                    }
                
                score_map[key]['weighted_score'] += weighted_score
                score_map[key]['component_scores'][f'list_{i}'] = weighted_score
        
        # Create fused list
        fused_results = []
        for key, data in score_map.items():
            result = data['result'].copy()
            result['weighted_score'] = data['weighted_score']
            result['component_scores'] = data['component_scores']
            fused_results.append(result)
        
        # Sort by weighted score
        fused_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        logger.info(f"  Fused {len(fused_results)} unique results")
        
        return fused_results
    
    def borda_count_fusion(self, ranked_lists: List[List[Dict]]) -> List[Dict[str, Any]]:
        """
        Borda count fusion.
        
        Each result gets points based on its position:
        points = len(list) - rank
        
        Args:
            ranked_lists: List of ranked result lists
            
        Returns:
            Fused ranked list
        """
        logger.info("Applying Borda Count Fusion")
        
        # Build score map
        score_map = {}
        
        for ranked_list in ranked_lists:
            list_len = len(ranked_list)
            
            for rank, result in enumerate(ranked_list):
                key = self._get_key(result)
                
                # Borda count: points decrease with rank
                points = list_len - rank
                
                if key not in score_map:
                    score_map[key] = {
                        'result': result,
                        'borda_score': 0
                    }
                
                score_map[key]['borda_score'] += points
        
        # Create fused list
        fused_results = []
        for key, data in score_map.items():
            result = data['result'].copy()
            result['borda_score'] = data['borda_score']
            fused_results.append(result)
        
        # Sort by Borda score
        fused_results.sort(key=lambda x: x['borda_score'], reverse=True)
        
        logger.info(f"  Fused {len(fused_results)} unique results")
        
        return fused_results
    
    def combine_with_metadata(self, results: List[Dict],
                             metadata_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Combine ranking with metadata scores.
        
        Args:
            results: Ranked results
            metadata_scores: Metadata-based scoring rules
                           e.g., {'source': {'textbook': 1.2, 'article': 1.0}}
            
        Returns:
            Reranked results with metadata boost
        """
        logger.info("Applying metadata-based reranking")
        
        for result in results:
            metadata_boost = 1.0
            
            # Apply metadata scoring rules
            for meta_field, scoring_rule in metadata_scores.items():
                if meta_field in result:
                    value = result[meta_field]
                    
                    if isinstance(scoring_rule, dict):
                        # Lookup-based scoring
                        metadata_boost *= scoring_rule.get(value, 1.0)
                    elif callable(scoring_rule):
                        # Function-based scoring
                        metadata_boost *= scoring_rule(value)
            
            # Apply boost to existing score
            base_score = result.get('hybrid_score', result.get('score', 0))
            result['final_score'] = base_score * metadata_boost
            result['metadata_boost'] = metadata_boost
        
        # Resort by final score
        results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        return results
    
    def _get_key(self, result: Dict) -> str:
        """Get unique key for result."""
        if 'chunk_id' in result:
            return f"chunk_{result['chunk_id']}"
        elif 'doc_id' in result:
            return f"doc_{result['doc_id']}"
        else:
            text = result.get('text', '')
            return f"text_{hash(text)}"
    
    def compare_fusion_methods(self, ranked_lists: List[List[Dict]],
                               ground_truth: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare different fusion methods.
        
        Args:
            ranked_lists: List of ranked result lists
            ground_truth: Optional ground truth ranking keys
            
        Returns:
            Comparison results
        """
        logger.info("Comparing fusion methods...")
        
        methods = {
            'rrf': self.reciprocal_rank_fusion(ranked_lists),
            'borda': self.borda_count_fusion(ranked_lists)
        }
        
        comparison = {
            'methods': {},
            'best_method': None
        }
        
        for method_name, fused_list in methods.items():
            stats = {
                'num_results': len(fused_list),
                'top_5_overlap': 0
            }
            
            if ground_truth:
                # Calculate metrics if ground truth provided
                top_5_keys = [self._get_key(r) for r in fused_list[:5]]
                stats['top_5_overlap'] = len(set(top_5_keys) & set(ground_truth[:5]))
            
            comparison['methods'][method_name] = stats
        
        # Determine best method
        if ground_truth:
            best = max(comparison['methods'].items(), 
                      key=lambda x: x[1]['top_5_overlap'])
            comparison['best_method'] = best[0]
        
        return comparison
