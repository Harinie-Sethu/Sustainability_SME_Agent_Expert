"""
Intelligent Context Assembly
Assemble retrieved chunks into coherent context for LLM
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextAssembler:
    """
    Intelligently assemble context from retrieved chunks:
    - Remove duplicates
    - Order by relevance or document structure
    - Add source citations
    - Manage token limits
    - Add diversity
    """
    
    def __init__(self, max_tokens: int = 4000, max_chunks: int = 10):
        """
        Initialize context assembler.
        
        Args:
            max_tokens: Maximum tokens for assembled context
            max_chunks: Maximum number of chunks
        """
        self.max_tokens = max_tokens
        self.max_chunks = max_chunks
        
        logger.info(f" Context Assembler initialized (max_tokens={max_tokens}, max_chunks={max_chunks})")
    
    def assemble_context(self, results: List[Dict], 
                        strategy: str = "relevance",
                        add_citations: bool = True,
                        add_metadata: bool = True,
                        diversity_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Assemble context from retrieval results.
        
        Args:
            results: Retrieved results
            strategy: Assembly strategy ("relevance", "document_order", "diversity")
            add_citations: Whether to add source citations
            add_metadata: Whether to include metadata
            diversity_threshold: Similarity threshold for diversity (0-1)
            
        Returns:
            Assembled context with metadata
        """
        logger.info(f"Assembling context from {len(results)} results (strategy={strategy})")
        
        # Apply strategy
        if strategy == "relevance":
            ordered_results = self._order_by_relevance(results)
        elif strategy == "document_order":
            ordered_results = self._order_by_document(results)
        elif strategy == "diversity":
            ordered_results = self._order_by_diversity(results, diversity_threshold)
        else:
            ordered_results = results
        
        # Assemble with token limit
        assembled = self._assemble_with_limit(
            ordered_results,
            add_citations=add_citations,
            add_metadata=add_metadata
        )
        
        return assembled
    
    def _order_by_relevance(self, results: List[Dict]) -> List[Dict]:
        """Order by relevance scores."""
        # Use hybrid_score if available, else fall back to other scores
        def get_score(result):
            return result.get('hybrid_score', 
                   result.get('final_score',
                   result.get('rerank_score',
                   result.get('score', 0))))
        
        return sorted(results, key=get_score, reverse=True)
    
    def _order_by_document(self, results: List[Dict]) -> List[Dict]:
        """Order by document and chunk position."""
        def get_sort_key(result):
            doc_id = result.get('doc_id', result.get('source_file', 'unknown'))
            chunk_id = result.get('chunk_id', 0)
            return (doc_id, chunk_id)
        
        return sorted(results, key=get_sort_key)
    
    def _order_by_diversity(self, results: List[Dict], threshold: float) -> List[Dict]:
        """
        Order with diversity - avoid too similar chunks.
        
        Args:
            results: Results to order
            threshold: Similarity threshold
            
        Returns:
            Diversified results
        """
        if not results:
            return []
        
        # Start with highest scoring result
        ordered = self._order_by_relevance(results)
        diverse_results = [ordered[0]]
        
        for result in ordered[1:]:
            # Check if sufficiently different from already selected
            if self._is_diverse(result, diverse_results, threshold):
                diverse_results.append(result)
            
            if len(diverse_results) >= self.max_chunks:
                break
        
        logger.info(f"  Diversified from {len(results)} to {len(diverse_results)} chunks")
        
        return diverse_results
    
    def _is_diverse(self, result: Dict, selected: List[Dict], threshold: float) -> bool:
        """
        Check if result is sufficiently diverse from selected results.
        
        Args:
            result: Result to check
            selected: Already selected results
            threshold: Diversity threshold
            
        Returns:
            True if diverse enough
        """
        result_text = result.get('text', '').lower()
        result_words = set(result_text.split())
        
        if not result_words:
            return False
        
        for selected_result in selected:
            selected_text = selected_result.get('text', '').lower()
            selected_words = set(selected_text.split())
            
            if not selected_words:
                continue
            
            # Calculate Jaccard similarity
            intersection = len(result_words & selected_words)
            union = len(result_words | selected_words)
            similarity = intersection / union if union > 0 else 0
            
            if similarity > threshold:
                return False  # Too similar
        
        return True
    
    def _assemble_with_limit(self, results: List[Dict],
                            add_citations: bool,
                            add_metadata: bool) -> Dict[str, Any]:
        """
        Assemble results respecting token limit.
        
        Args:
            results: Ordered results
            add_citations: Add source citations
            add_metadata: Add metadata info
            
        Returns:
            Assembled context
        """
        context_parts = []
        included_chunks = []
        total_tokens = 0
        
        for i, result in enumerate(results[:self.max_chunks]):
            # Estimate tokens (rough: 1 token ≈ 4 characters)
            text = result.get('text', '')
            chunk_tokens = len(text) // 4
            
            # Check if adding this chunk would exceed limit
            if total_tokens + chunk_tokens > self.max_tokens and i > 0:
                logger.info(f"  Stopped at {i} chunks due to token limit")
                break
            
            # Build chunk text with citation
            chunk_text = ""
            
            if add_citations:
                source = result.get('source_file', 'Unknown')
                score = result.get('hybrid_score', result.get('score', 0))
                chunk_text += f"[Source {i+1}: {source}] (Relevance: {score:.2f})\n"
            
            if add_metadata:
                metadata_parts = []
                if 'subject' in result:
                    metadata_parts.append(f"Subject: {result['subject']}")
                if 'doc_type' in result:
                    metadata_parts.append(f"Type: {result['doc_type']}")
                
                if metadata_parts:
                    chunk_text += f"({', '.join(metadata_parts)})\n"
            
            chunk_text += text + "\n"
            
            context_parts.append(chunk_text)
            included_chunks.append(result)
            total_tokens += chunk_tokens
        
        # Join all parts
        assembled_text = "\n".join(context_parts)
        
        return {
            'context': assembled_text,
            'num_chunks': len(included_chunks),
            'total_tokens': total_tokens,
            'chunks': included_chunks,
            'truncated': len(results) > len(included_chunks)
        }
    
    def create_structured_context(self, results: List[Dict],
                                  query: str) -> str:
        """
        Create structured context with sections.
        
        Args:
            results: Retrieved results
            query: Original query
            
        Returns:
            Structured context string
        """
        assembled = self.assemble_context(results)
        
        structured = f"""Query: {query}

Retrieved Information:

{assembled['context']}

Summary:
- Retrieved {assembled['num_chunks']} relevant chunks
- Total context: ~{assembled['total_tokens']} tokens
- Sources cover information about the query
"""
        
        return structured
    
    def get_source_diversity(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze source diversity in results.
        
        Args:
            results: Results to analyze
            
        Returns:
            Diversity metrics
        """
        sources = {}
        doc_types = {}
        subjects = {}
        
        for result in results:
            source = result.get('source_file', 'unknown')
            sources[source] = sources.get(source, 0) + 1
            
            doc_type = result.get('doc_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            subject = result.get('subject', 'unknown')
            subjects[subject] = subjects.get(subject, 0) + 1
        
        return {
            'unique_sources': len(sources),
            'source_distribution': sources,
            'unique_doc_types': len(doc_types),
            'doc_type_distribution': doc_types,
            'unique_subjects': len(subjects),
            'subject_distribution': subjects
        }
