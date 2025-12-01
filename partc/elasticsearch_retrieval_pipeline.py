#!/usr/bin/env python3
"""
Complete Retrieval Pipeline for Environment/Sustainability SME
Integrates vector search, parent-child retrieval, and reranking.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

# Handle different Elasticsearch versions
try:
    from elasticsearch import Elasticsearch
    try:
        from elasticsearch.exceptions import ElasticsearchException
    except ImportError:
        ElasticsearchException = Exception  # Fallback
except ImportError:
    print("Please install: pip install elasticsearch")
    import sys
    sys.exit(1)

from sentence_transformers import SentenceTransformer
import torch
from reranker import BGEReranker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retrieval.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Complete retrieval pipeline with hierarchical search and reranking."""
    
    def __init__(self, es_host: str = "localhost", es_port: int = 9200,
                 embedding_model: str = "all-mpnet-base-v2"):
        """
        Initialize the retrieval pipeline.
        
        Args:
            es_host: Elasticsearch host
            es_port: Elasticsearch port
            embedding_model: Model name for query encoding
        """
        # Initialize Elasticsearch client
        try:
            self.es_client = Elasticsearch(
                [{'host': es_host, 'port': es_port, 'scheme': 'http'}],
                request_timeout=30
            )
            
            if not self.es_client.ping():
                raise ConnectionError("Could not connect to Elasticsearch")
            
            logger.info(f"Connected to Elasticsearch at {es_host}:{es_port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {str(e)}")
            raise
        
        # Initialize embedding model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model: {embedding_model} on {self.device}")
        
        self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
        self.embedding_field = "embedding_mpnet"  # Default to mpnet embeddings
        
        # Initialize reranker
        self.reranker = BGEReranker()
        
        # Index names
        self.parent_index = "environment_documents"
        self.chunk_index = "environment_chunks"
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode query into embedding vector.
        
        Args:
            query: Query string
            
        Returns:
            Query embedding vector
        """
        try:
            embedding = self.embedding_model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding query: {str(e)}")
            raise
    
    def vector_search(self, query_embedding: np.ndarray, 
                     top_k: int = 20, 
                     filters: Dict = None) -> List[Dict]:
        """
        Perform vector similarity search on chunks.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Additional filters (optional)
            
        Returns:
            List of retrieved chunks
        """
        try:
            # Build query
            query_body = {
                "size": top_k,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": f"cosineSimilarity(params.query_vector, '{self.embedding_field}') + 1.0",
                            "params": {"query_vector": query_embedding.tolist()}
                        }
                    }
                },
                "_source": {
                    "excludes": [self.embedding_field]  # Don't return embeddings in results
                }
            }
            
            # Add filters if provided
            if filters:
                query_body["query"]["script_score"]["query"] = {
                    "bool": {
                        "must": [{"match_all": {}}],
                        "filter": [filters]
                    }
                }
            
            # Execute search
            response = self.es_client.search(
                index=self.chunk_index,
                body=query_body
            )
            
            # Extract results
            results = []
            for hit in response['hits']['hits']:
                result = {
                    "chunk_id": hit['_source']['chunk_id'],
                    "parent_document_id": hit['_source']['parent_document_id'],
                    "text": hit['_source']['text'],
                    "metadata": hit['_source']['metadata'],
                    "score": hit['_score'] - 1.0,  # Subtract 1.0 to get actual cosine similarity
                    "es_id": hit['_id']
                }
                results.append(result)
            
            logger.info(f"Vector search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error during vector search: {str(e)}")
            return []
    
    def get_parent_documents(self, parent_ids: List[str]) -> Dict[str, Dict]:
        """
        Retrieve parent document metadata for given IDs.
        
        Args:
            parent_ids: List of parent document IDs
            
        Returns:
            Dictionary mapping parent IDs to their metadata
        """
        try:
            # Query for parent documents
            query_body = {
                "query": {
                    "terms": {
                        "document_id": parent_ids
                    }
                }
            }
            
            response = self.es_client.search(
                index=self.parent_index,
                body=query_body,
                size=len(parent_ids)
            )
            
            # Build parent metadata map
            parent_map = {}
            for hit in response['hits']['hits']:
                doc_id = hit['_source']['document_id']
                parent_map[doc_id] = hit['_source']
            
            return parent_map
            
        except Exception as e:
            logger.error(f"Error retrieving parent documents: {str(e)}")
            return {}
    
    def get_sibling_chunks(self, parent_id: str, current_chunk_id: int,
                          window_size: int = 2) -> List[Dict]:
        """
        Retrieve sibling chunks from the same parent document.
        
        Args:
            parent_id: Parent document ID
            current_chunk_id: Current chunk ID
            window_size: Number of chunks before and after to retrieve
            
        Returns:
            List of sibling chunks
        """
        try:
            # Query for sibling chunks
            query_body = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"parent_document_id": parent_id}},
                            {
                                "range": {
                                    "chunk_id": {
                                        "gte": max(0, current_chunk_id - window_size),
                                        "lte": current_chunk_id + window_size
                                    }
                                }
                            }
                        ]
                    }
                },
                "sort": [{"chunk_id": "asc"}],
                "size": window_size * 2 + 1,
                "_source": {
                    "excludes": [self.embedding_field, "embedding_specter"]
                }
            }
            
            response = self.es_client.search(
                index=self.chunk_index,
                body=query_body
            )
            
            siblings = [hit['_source'] for hit in response['hits']['hits']]
            return siblings
            
        except Exception as e:
            logger.error(f"Error retrieving sibling chunks: {str(e)}")
            return []
    
    def hierarchical_retrieve(self, query: str, 
                            top_k: int = 10,
                            expand_context: bool = True,
                            rerank: bool = True) -> Dict:
        """
        Perform hierarchical retrieval with optional context expansion and reranking.
        
        Args:
            query: Query string
            top_k: Number of final results to return
            expand_context: Whether to expand with sibling chunks
            rerank: Whether to apply reranking
            
        Returns:
            Dictionary with retrieval results and metadata
        """
        logger.info(f"Hierarchical retrieval for query: '{query}'")
        
        # Step 1: Encode query
        query_embedding = self.encode_query(query)
        
        # Step 2: Initial vector search (retrieve more for reranking)
        initial_k = top_k * 2 if rerank else top_k
        initial_results = self.vector_search(query_embedding, top_k=initial_k)
        
        if not initial_results:
            logger.warning("No results found for query")
            return {
                "query": query,
                "results": [],
                "metadata": {"total_results": 0}
            }
        
        # Step 3: Rerank if enabled
        if rerank:
            logger.info("Applying reranking...")
            reranked_results = self.reranker.rerank(
                query, 
                initial_results,
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            # Calculate reranking metrics
            rerank_metrics = self.reranker.calculate_relevance_metrics(reranked_results)
            logger.info(f"Reranking metrics: {rerank_metrics}")
        else:
            reranked_results = initial_results[:top_k]
            rerank_metrics = {}
        
        # Step 4: Expand with context if enabled
        if expand_context:
            logger.info("Expanding with contextual chunks...")
            for result in reranked_results:
                siblings = self.get_sibling_chunks(
                    result['parent_document_id'],
                    result['chunk_id'],
                    window_size=1
                )
                result['context_chunks'] = siblings
        
        # Step 5: Retrieve parent metadata
        parent_ids = list(set(r['parent_document_id'] for r in reranked_results))
        parent_metadata = self.get_parent_documents(parent_ids)
        
        # Attach parent metadata to results
        for result in reranked_results:
            result['parent_metadata'] = parent_metadata.get(
                result['parent_document_id'], {}
            )
        
        # Prepare final response
        response = {
            "query": query,
            "results": reranked_results,
            "metadata": {
                "total_results": len(reranked_results),
                "initial_results": len(initial_results),
                "reranking_applied": rerank,
                "context_expanded": expand_context,
                "unique_documents": len(parent_ids),
                "rerank_metrics": rerank_metrics
            }
        }
        
        logger.info(f"Retrieved {len(reranked_results)} results from {len(parent_ids)} documents")
        return response
    
    def hybrid_search(self, query: str, top_k: int = 10, 
                     alpha: float = 0.7) -> Dict:
        """
        Perform hybrid search combining vector and keyword search.
        
        Args:
            query: Query string
            top_k: Number of results to return
            alpha: Weight for vector search (1-alpha for keyword search)
            
        Returns:
            Dictionary with hybrid search results
        """
        logger.info(f"Hybrid search for query: '{query}'")
        
        # Step 1: Vector search
        query_embedding = self.encode_query(query)
        vector_results = self.vector_search(query_embedding, top_k=top_k * 2)
        
        # Step 2: Keyword search
        try:
            keyword_query = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text^2", "parent_document_id"],
                        "type": "best_fields"
                    }
                },
                "size": top_k * 2,
                "_source": {
                    "excludes": [self.embedding_field, "embedding_specter"]
                }
            }
            
            keyword_response = self.es_client.search(
                index=self.chunk_index,
                body=keyword_query
            )
            
            keyword_results = []
            for hit in keyword_response['hits']['hits']:
                result = {
                    "chunk_id": hit['_source']['chunk_id'],
                    "parent_document_id": hit['_source']['parent_document_id'],
                    "text": hit['_source']['text'],
                    "metadata": hit['_source']['metadata'],
                    "score": hit['_score'],
                    "es_id": hit['_id']
                }
                keyword_results.append(result)
            
        except Exception as e:
            logger.error(f"Error during keyword search: {str(e)}")
            keyword_results = []
        
        # Step 3: Combine results with weighted scoring
        combined_scores = {}
        
        # Normalize and weight vector scores
        if vector_results:
            max_vector_score = max(r['score'] for r in vector_results)
            for result in vector_results:
                es_id = result['es_id']
                normalized_score = result['score'] / max_vector_score if max_vector_score > 0 else 0
                combined_scores[es_id] = {
                    'result': result,
                    'score': alpha * normalized_score
                }
        
        # Normalize and weight keyword scores
        if keyword_results:
            max_keyword_score = max(r['score'] for r in keyword_results)
            for result in keyword_results:
                es_id = result['es_id']
                normalized_score = result['score'] / max_keyword_score if max_keyword_score > 0 else 0
                
                if es_id in combined_scores:
                    combined_scores[es_id]['score'] += (1 - alpha) * normalized_score
                else:
                    combined_scores[es_id] = {
                        'result': result,
                        'score': (1 - alpha) * normalized_score
                    }
        
        # Sort by combined score and take top k
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:top_k]
        
        final_results = [item['result'] for item in sorted_results]
        for idx, item in enumerate(sorted_results):
            final_results[idx]['hybrid_score'] = item['score']
        
        # Get parent metadata
        parent_ids = list(set(r['parent_document_id'] for r in final_results))
        parent_metadata = self.get_parent_documents(parent_ids)
        
        for result in final_results:
            result['parent_metadata'] = parent_metadata.get(
                result['parent_document_id'], {}
            )
        
        response = {
            "query": query,
            "results": final_results,
            "metadata": {
                "total_results": len(final_results),
                "search_type": "hybrid",
                "alpha": alpha,
                "vector_results": len(vector_results),
                "keyword_results": len(keyword_results),
                "unique_documents": len(parent_ids)
            }
        }
        
        logger.info(f"Hybrid search returned {len(final_results)} results")
        return response
    
    def semantic_search_with_filters(self, query: str, 
                                    filters: Dict = None,
                                    top_k: int = 10) -> Dict:
        """
        Semantic search with metadata filters.
        
        Args:
            query: Query string
            filters: Metadata filters
            top_k: Number of results
            
        Returns:
            Filtered search results
        """
        query_embedding = self.encode_query(query)
        results = self.vector_search(query_embedding, top_k=top_k, filters=filters)
        
        return {
            "query": query,
            "results": results,
            "filters": filters,
            "metadata": {
                "total_results": len(results)
            }
        }


def main():
    """Main function to demonstrate retrieval pipeline."""
    # Initialize pipeline
    pipeline = RetrievalPipeline()
    
    # Example queries
    test_queries = [
        "What are the main causes of climate change?",
        "How does deforestation affect biodiversity?",
        "What are sustainable development practices?",
        "Explain the greenhouse effect"
    ]
    
    logger.info("="*60)
    logger.info("Testing Retrieval Pipeline")
    logger.info("="*60)
    
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        logger.info("-"*60)
        
        # Hierarchical retrieval with reranking
        results = pipeline.hierarchical_retrieve(
            query,
            top_k=5,
            expand_context=True,
            rerank=True
        )
        
        logger.info(f"Retrieved {results['metadata']['total_results']} results")
        logger.info(f"From {results['metadata']['unique_documents']} unique documents")
        
        # Display top result
        if results['results']:
            top_result = results['results'][0]
            logger.info(f"\nTop Result:")
            logger.info(f"  Score: {top_result.get('rerank_score', top_result.get('score')):.4f}")
            logger.info(f"  Document: {top_result['parent_metadata'].get('source_file', 'Unknown')}")
            logger.info(f"  Text preview: {top_result['text'][:200]}...")
        
        logger.info("\n")


if __name__ == "__main__":
    main()
