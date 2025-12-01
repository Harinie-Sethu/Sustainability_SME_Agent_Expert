"""
Hybrid Retrieval System
Combines dense vector search with sparse BM25 retrieval
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np

from partg.bm25_retriever import BM25Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retrieval combining:
    - Dense vector search (semantic similarity)
    - Sparse BM25 search (keyword matching)
    - Intelligent fusion of results
    """
    
    def __init__(self, vector_retriever, bm25_retriever: Optional[BM25Retriever] = None,
                 alpha: float = 0.5):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_retriever: Dense vector retriever (e.g., MilvusRetrievalPipeline)
            bm25_retriever: BM25 retriever (creates new if None)
            alpha: Weight for dense retrieval (0-1), (1-alpha) for sparse
                  alpha=1.0: only dense, alpha=0.0: only sparse, alpha=0.5: equal weight
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.alpha = alpha
        
        self._bm25_indexed = False
        
        logger.info(f"✓ Hybrid Retriever initialized (alpha={alpha})")
    
    def index_documents_for_bm25(self, documents: List[Dict[str, Any]], 
                                 text_field: str = 'text'):
        """
        Index documents for BM25 retrieval.
        
        Args:
            documents: Documents to index
            text_field: Field containing text
        """
        self.bm25_retriever.index_documents(documents, text_field)
        self._bm25_indexed = True
    
    def is_bm25_indexed(self) -> bool:
        """Check if BM25 is indexed."""
        return self._bm25_indexed or self.bm25_retriever.is_indexed
    
    def hybrid_retrieve(self, query: str, top_k: int = 10,
                       dense_k: Optional[int] = None,
                       sparse_k: Optional[int] = None,
                       rerank: bool = True) -> Dict[str, Any]:
        """
        Perform hybrid retrieval.
        
        Args:
            query: Query text
            top_k: Final number of results
            dense_k: Number of results from dense retrieval (default: 2*top_k)
            sparse_k: Number of results from sparse retrieval (default: 2*top_k)
            rerank: Whether to apply reranking
            
        Returns:
            Hybrid retrieval results
        """
        logger.info(f"Hybrid retrieval: '{query}' (top_k={top_k})")
        
        # Default: retrieve more candidates for fusion
        if dense_k is None:
            dense_k = top_k * 2
        if sparse_k is None:
            sparse_k = top_k * 2
        
        # 1. Dense vector retrieval
        logger.info(f"  Dense retrieval (k={dense_k})...")
        dense_results = self._get_dense_results(query, dense_k, rerank)
        
        # 2. Sparse BM25 retrieval
        logger.info(f"  Sparse BM25 retrieval (k={sparse_k})...")
        sparse_results = self._get_sparse_results(query, sparse_k)
        
        # 3. Fuse results
        logger.info(f"  Fusing results...")
        fused_results = self._fuse_results(dense_results, sparse_results, top_k)
        
        return {
            'query': query,
            'results': fused_results[:top_k],
            'num_dense': len(dense_results),
            'num_sparse': len(sparse_results),
            'num_fused': len(fused_results),
            'alpha': self.alpha
        }
    
    def _get_dense_results(self, query: str, k: int, rerank: bool) -> List[Dict[str, Any]]:
        """Get results from dense vector retrieval."""
        try:
            # Use existing vector retriever
            results = self.vector_retriever.hierarchical_retrieve(
                query,
                top_k=k,
                expand_context=False,
                rerank=rerank
            )
            
            # Normalize dense results format
            dense_results = []
            for result in results.get('results', []):
                dense_results.append({
                    **result,
                    'dense_score': result.get('rerank_score', result.get('score', 0)),
                    'retrieval_method': 'dense'
                })
            
            return dense_results
        
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return []
    
    def _get_sparse_results(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Get results from sparse BM25 retrieval."""
        if not self.is_bm25_indexed():
            logger.warning("BM25 not indexed, skipping sparse retrieval")
            return []
        
        try:
            sparse_results = self.bm25_retriever.retrieve(query, top_k=k)
            
            # Normalize sparse results format
            for result in sparse_results:
                result['sparse_score'] = result.get('bm25_score', 0)
                result['retrieval_method'] = 'sparse'
            
            return sparse_results
        
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}")
            return []
    
    def _fuse_results(self, dense_results: List[Dict], sparse_results: List[Dict],
                     top_k: int) -> List[Dict[str, Any]]:
        """
        Fuse dense and sparse results using weighted combination.
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            top_k: Number of final results
            
        Returns:
            Fused results
        """
        # Create lookup for results by chunk_id or text
        result_map = {}
        
        # Normalize dense scores to [0, 1]
        if dense_results:
            max_dense = max(r.get('dense_score', 0) for r in dense_results)
            min_dense = min(r.get('dense_score', 0) for r in dense_results)
            dense_range = max_dense - min_dense if max_dense > min_dense else 1.0
            
            for result in dense_results:
                key = self._get_result_key(result)
                normalized_dense = (result.get('dense_score', 0) - min_dense) / dense_range
                
                result_map[key] = {
                    **result,
                    'normalized_dense': normalized_dense,
                    'normalized_sparse': 0.0
                }
        
        # Normalize sparse scores to [0, 1]
        if sparse_results:
            max_sparse = max(r.get('sparse_score', 0) for r in sparse_results)
            min_sparse = min(r.get('sparse_score', 0) for r in sparse_results)
            sparse_range = max_sparse - min_sparse if max_sparse > min_sparse else 1.0
            
            for result in sparse_results:
                key = self._get_result_key(result)
                normalized_sparse = (result.get('sparse_score', 0) - min_sparse) / sparse_range
                
                if key in result_map:
                    # Document found in both retrievals
                    result_map[key]['normalized_sparse'] = normalized_sparse
                    result_map[key]['in_both'] = True
                else:
                    # Document only in sparse results
                    result_map[key] = {
                        **result,
                        'normalized_dense': 0.0,
                        'normalized_sparse': normalized_sparse,
                        'in_both': False
                    }
        
        # Calculate hybrid scores
        fused_results = []
        for key, result in result_map.items():
            # Hybrid score: weighted combination
            hybrid_score = (
                self.alpha * result['normalized_dense'] +
                (1 - self.alpha) * result['normalized_sparse']
            )
            
            # Bonus for appearing in both
            if result.get('in_both', False):
                hybrid_score *= 1.1  # 10% bonus
            
            result['hybrid_score'] = hybrid_score
            fused_results.append(result)
        
        # Sort by hybrid score
        fused_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        logger.info(f"  Fused {len(fused_results)} unique results")
        logger.info(f"  Top hybrid score: {fused_results[0]['hybrid_score']:.3f}" if fused_results else "  No results")
        
        return fused_results
    
    def _get_result_key(self, result: Dict) -> str:
        """
        Get unique key for a result.
        
        Args:
            result: Result dictionary
            
        Returns:
            Unique key
        """
        # Try to use chunk_id or doc_id
        if 'chunk_id' in result:
            return f"chunk_{result['chunk_id']}"
        elif 'doc_id' in result:
            return f"doc_{result['doc_id']}"
        else:
            # Fallback: use hash of text
            text = result.get('text', '')
            return f"text_{hash(text)}"
    
    def get_retrieval_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics about retrieval."""
        fused = results.get('results', [])
        
        stats = {
            'total_results': len(fused),
            'from_dense_only': 0,
            'from_sparse_only': 0,
            'from_both': 0,
            'avg_hybrid_score': 0.0,
            'score_distribution': {
                'dense': [],
                'sparse': [],
                'hybrid': []
            }
        }
        
        for result in fused:
            if result.get('in_both', False):
                stats['from_both'] += 1
            elif result.get('normalized_dense', 0) > 0:
                stats['from_dense_only'] += 1
            else:
                stats['from_sparse_only'] += 1
            
            stats['score_distribution']['dense'].append(result.get('normalized_dense', 0))
            stats['score_distribution']['sparse'].append(result.get('normalized_sparse', 0))
            stats['score_distribution']['hybrid'].append(result.get('hybrid_score', 0))
        
        if fused:
            stats['avg_hybrid_score'] = np.mean(stats['score_distribution']['hybrid'])
        
        return stats
            stats['avg_hybrid_score'] = np.mean(stats['score_distribution']['hybrid'])
        
        return stats

