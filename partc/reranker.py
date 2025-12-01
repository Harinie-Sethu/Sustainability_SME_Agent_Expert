#!/usr/bin/env python3
"""
Reranking Module for Environment/Sustainability Documents
Implements BGE Reranker with fallback to similarity search.
"""

import logging
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import CrossEncoder
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BGEReranker:
    """Reranking using BGE Reranker with fallback to cosine similarity."""
    
    def __init__(self, model_name: str = 'BAAI/bge-reranker-base'):
        """
        Initialize the reranker.
        
        Args:
            model_name: Name of the cross-encoder model
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing BGE Reranker on {self.device}")
        
        try:
            self.reranker = CrossEncoder(model_name, device=self.device)
            self.reranker_available = True
            logger.info(f"Successfully loaded BGE Reranker: {model_name}")
            
        except Exception as e:
            logger.warning(f"Could not load BGE Reranker: {str(e)}")
            logger.warning("Will fallback to cosine similarity for reranking")
            self.reranker_available = False
    
    def rerank_with_bge(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Rerank candidates using BGE Reranker.
        
        Args:
            query: Query string
            candidates: List of candidate dictionaries with 'text' and 'score' fields
            
        Returns:
            Reranked list of candidates
        """
        if not candidates:
            return []
        
        try:
            # Prepare query-document pairs
            pairs = [[query, candidate['text']] for candidate in candidates]
            
            # Get reranking scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Update candidates with rerank scores
            for idx, candidate in enumerate(candidates):
                candidate['rerank_score'] = float(rerank_scores[idx])
                candidate['original_score'] = candidate.get('score', 0.0)
            
            # Sort by rerank score
            reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
            
            logger.info(f"Reranked {len(candidates)} candidates using BGE Reranker")
            return reranked
            
        except Exception as e:
            logger.error(f"Error during BGE reranking: {str(e)}")
            logger.info("Falling back to original scores")
            return candidates
    
    def rerank_with_cosine(self, query_embedding: np.ndarray, 
                           candidates: List[Dict]) -> List[Dict]:
        """
        Fallback reranking using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            candidates: List of candidate dictionaries with 'embedding' field
            
        Returns:
            Reranked list of candidates
        """
        if not candidates:
            return []
        
        try:
            # Calculate cosine similarities
            for candidate in candidates:
                if 'embedding' in candidate:
                    candidate_embedding = np.array(candidate['embedding'])
                    
                    # Cosine similarity
                    similarity = np.dot(query_embedding, candidate_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding)
                    )
                    
                    candidate['rerank_score'] = float(similarity)
                    candidate['original_score'] = candidate.get('score', 0.0)
            
            # Sort by rerank score
            reranked = sorted(candidates, key=lambda x: x.get('rerank_score', 0.0), reverse=True)
            
            logger.info(f"Reranked {len(candidates)} candidates using cosine similarity")
            return reranked
            
        except Exception as e:
            logger.error(f"Error during cosine reranking: {str(e)}")
            return candidates
    
    def rerank(self, query: str, candidates: List[Dict], 
               query_embedding: np.ndarray = None, top_k: int = None) -> List[Dict]:
        """
        Rerank candidates with fallback strategy.
        
        Args:
            query: Query string
            candidates: List of candidate dictionaries
            query_embedding: Query embedding for fallback (optional)
            top_k: Return top k results (optional)
            
        Returns:
            Reranked list of candidates
        """
        if not candidates:
            return []
        
        # Try BGE reranker first
        if self.reranker_available:
            reranked = self.rerank_with_bge(query, candidates)
        
        # Fallback to cosine similarity
        elif query_embedding is not None:
            reranked = self.rerank_with_cosine(query_embedding, candidates)
        
        # No reranking possible
        else:
            logger.warning("No reranking method available, returning original order")
            reranked = candidates
        
        # Return top k if specified
        if top_k is not None and top_k > 0:
            reranked = reranked[:top_k]
        
        return reranked
    
    def calculate_relevance_metrics(self, reranked_results: List[Dict]) -> Dict:
        """
        Calculate relevance metrics for reranked results.
        
        Args:
            reranked_results: List of reranked candidates
            
        Returns:
            Dictionary of metrics
        """
        if not reranked_results:
            return {}
        
        metrics = {
            "total_results": len(reranked_results),
            "avg_rerank_score": np.mean([r.get('rerank_score', 0.0) for r in reranked_results]),
            "avg_original_score": np.mean([r.get('original_score', 0.0) for r in reranked_results]),
            "score_improvement": 0.0
        }
        
        # Calculate score improvement
        if metrics["avg_original_score"] > 0:
            metrics["score_improvement"] = (
                (metrics["avg_rerank_score"] - metrics["avg_original_score"]) / 
                metrics["avg_original_score"] * 100
            )
        
        return metrics
