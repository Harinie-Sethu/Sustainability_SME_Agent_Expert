#!/usr/bin/env python3
"""
Milvus Retrieval Pipeline for Environment/Sustainability SME
Integrates Milvus vector search, parent-child retrieval, and reranking.
"""

# CRITICAL: Set protobuf implementation BEFORE any other imports
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Fix for pymilvus 2.3.3 requiring runtime_version (not in protobuf 3.20.3)
import sys
try:
    from google.protobuf import runtime_version
except ImportError:
    # Monkey patch runtime_version for protobuf 3.20.3 compatibility
    from unittest.mock import MagicMock
    class MockRuntimeVersion:
        Domain = MagicMock()
        ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
    sys.modules['google.protobuf.runtime_version'] = MockRuntimeVersion()

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer
import torch

# Support execution both as package and standalone script
try:
    from .reranker import BGEReranker  # type: ignore
except ImportError:  # pragma: no cover
    from reranker import BGEReranker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('milvus_retrieval.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MilvusRetrievalPipeline:
    """Complete retrieval pipeline with Milvus vector search and reranking."""
    
    def __init__(self, db_path: str, embedding_model: str = "all-mpnet-base-v2", 
                 model_key: str = None, prefer_scientific: bool = True):
        """
        Initialize the Milvus retrieval pipeline.
        
        Args:
            db_path: Path to Milvus database file (or directory for Milvus Lite)
            embedding_model: Model name for query encoding (should match indexed model)
            model_key: Key for the embedding model ('baseline' or 'scientific')
            prefer_scientific: If True and model_key not specified, prefer 'scientific' for environment domain
        """
        # Connect to Milvus
        # Try to connect to existing Milvus server (standalone or Lite)
        try:
            # First try localhost:19530 (standard Milvus port)
            connections.connect(alias="default", host="localhost", port="19530", timeout=10)
            logger.info("Connected to Milvus server on localhost:19530")
        except Exception as e1:
            logger.warning(f"Could not connect to Milvus server on localhost:19530: {e1}")
            # Try URI-based connection (pymilvus 2.6+) - same as indexer
            try:
                logger.info("Attempting URI-based connection (pymilvus 2.6+)...")
                from pathlib import Path
                
                # Use same URI path as indexer - use absolute path
                db_path_obj = Path(db_path)
                if db_path_obj.is_file():
                    db_dir = db_path_obj.parent
                    uri_path = str(db_path_obj.resolve())  # Use the actual file path
                else:
                    db_dir = db_path_obj
                    db_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                    uri_path = str((db_dir / "milvus_lite.db").resolve())  # Absolute path
                
                # Disconnect any existing connection first
                try:
                    connections.disconnect("default")
                except:
                    pass
                
                # Try different URI formats
                try:
                    connections.connect(alias="default", uri=uri_path)
                    logger.info(f"Connected to Milvus Lite via URI: {uri_path}")
                except Exception as e1:
                    # Try with file:// prefix
                    try:
                        connections.connect(alias="default", uri=f"file://{uri_path}")
                        logger.info(f"Connected to Milvus Lite via URI (file://): {uri_path}")
                    except Exception as e2:
                        raise Exception(f"Both URI formats failed: {e1}, {e2}")
            except Exception as uri_error:
                logger.warning(f"URI connection failed: {uri_error}")
                # Try starting Milvus Lite server as last resort
                try:
                    logger.info("Attempting to start Milvus Lite server...")
                    import time
                    
                    # Try different import paths for Milvus Lite
                    default_server = None
                    try:
                        from pymilvus import MilvusServer
                        default_server = MilvusServer()
                    except ImportError:
                        try:
                            from milvus import default_server
                        except ImportError:
                            try:
                                from pymilvus.server import default_server
                            except ImportError:
                                raise ImportError(
                                    "Milvus Lite not available. Install with: pip install 'pymilvus[milvus_lite]' or pip install milvus-lite"
                                )
                    
                    # Use parent directory of db_path as data directory
                    db_dir = Path(db_path).parent if Path(db_path).is_file() else Path(db_path)
                    data_dir = str(db_dir / "milvus_lite_data")
                    
                    # Start Milvus Lite server if not running
                    if not default_server.running:
                        default_server.set_base_dir(data_dir)
                        default_server.start()
                        logger.info(f"Started Milvus Lite server with data directory: {data_dir}")
                        # Wait for server to be ready (with timeout)
                        max_wait = 30
                        waited = 0
                        while waited < max_wait and not default_server.running:
                            time.sleep(1)
                            waited += 1
                        if not default_server.running:
                            raise TimeoutError("Milvus Lite server did not start in time")
                    
                    # Connect to the local server
                    connections.connect(alias="default", host="localhost", port="19530", timeout=10)
                    logger.info("Connected to Milvus Lite server on localhost:19530")
                except ImportError as e2:
                    logger.error(f"Milvus Lite not installed: {e2}")
                    logger.error("Please install Milvus Lite with one of these commands:")
                    logger.error("  pip install 'pymilvus[milvus_lite]'")
                    logger.error("  OR")
                    logger.error("  pip install milvus-lite")
                    raise ConnectionError(
                        "Milvus Lite not installed. Install with: pip install 'pymilvus[milvus_lite]' or pip install milvus-lite"
                    )
                except Exception as e2:
                    logger.error(f"Failed to start/connect to Milvus Lite: {e2}")
                    raise ConnectionError(f"Could not connect to Milvus server. Please ensure Milvus is running or install milvus-lite. Error: {e2}")
        
        # Determine model key
        if model_key:
            self.model_key = model_key
        elif prefer_scientific:
            # For environment/sustainability domain, prefer scientific embeddings
            self.model_key = "scientific"
            if embedding_model == "all-mpnet-base-v2":
                embedding_model = "allenai/scibert_scivocab_uncased"
                logger.info("Using scientific embeddings (scibert) for environment domain")
        else:
            self.model_key = "baseline"
        
        # Initialize embedding model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model: {embedding_model} on {self.device}")
        logger.info(f"Using model key: {self.model_key} for environment/sustainability domain")
        
        self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
        
        # Initialize reranker
        self.reranker = BGEReranker()
        
        # Collection names
        self.parent_collection_name = "environment_documents"
        self.chunk_collection_name = f"environment_chunks_{self.model_key}"
        
        # Check available collections and use the best one
        self.milvus_available = False
        try:
            from pymilvus import utility
            
            # Check if collections exist
            available_collections = utility.list_collections()
            logger.info(f"Available collections: {available_collections}")
            
            # Try to load the preferred collection
            if self.chunk_collection_name in available_collections:
                self.chunk_collection = Collection(self.chunk_collection_name)
                self.chunk_collection.load()
                self.milvus_available = True
                logger.info(f"✓ Loaded collection: {self.chunk_collection_name}")
                
                # Check if collection has data
                num_entities = self.chunk_collection.num_entities
                logger.info(f"Collection has {num_entities} chunks indexed")
                
                if num_entities == 0:
                    logger.warning(f"Collection {self.chunk_collection_name} is empty!")
                    # Try fallback to baseline if scientific is empty
                    if self.model_key == "scientific" and "environment_chunks_baseline" in available_collections:
                        logger.info("Falling back to baseline collection")
                        self.model_key = "baseline"
                        self.chunk_collection_name = f"environment_chunks_{self.model_key}"
                        self.chunk_collection = Collection(self.chunk_collection_name)
                        self.chunk_collection.load()
                        num_entities = self.chunk_collection.num_entities
                        logger.info(f"Baseline collection has {num_entities} chunks")
            else:
                # Try fallback to baseline
                baseline_collection = "environment_chunks_baseline"
                if baseline_collection in available_collections:
                    logger.warning(f"Preferred collection {self.chunk_collection_name} not found, using {baseline_collection}")
                    self.model_key = "baseline"
                    self.chunk_collection_name = baseline_collection
                    self.chunk_collection = Collection(self.chunk_collection_name)
                    self.chunk_collection.load()
                    self.milvus_available = True
                    logger.info(f"✓ Loaded fallback collection: {baseline_collection}")
                else:
                    raise Exception(f"Neither {self.chunk_collection_name} nor {baseline_collection} found")
            
            # Load parent collection
            if self.parent_collection_name in available_collections:
                self.parent_collection = Collection(self.parent_collection_name)
                self.parent_collection.load()
                logger.info(f"✓ Loaded parent collection: {self.parent_collection_name}")
            else:
                logger.warning(f"Parent collection {self.parent_collection_name} not found")
                self.parent_collection = None
                
        except Exception as e:
            logger.error(f"Failed to load collections: {str(e)}")
            logger.error("RAG will fall back to web search only")
            self.milvus_available = False
            self.chunk_collection = None
            self.parent_collection = None
    
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
    
    def vector_search(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Dict]:
        """
        Perform vector similarity search on chunks.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of retrieved chunks
        """
        if not self.milvus_available or self.chunk_collection is None:
            logger.warning("Milvus not available, returning empty results")
            return []
        
        try:
            # Search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # Perform search
            results = self.chunk_collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["chunk_id", "parent_document_id", "text", "source_file", 
                             "chunk_size_tokens", "estimated_tokens"]
            )
            
            # Parse results
            retrieved_chunks = []
            for hits in results:
                for hit in hits:
                    chunk = {
                        "chunk_id": hit.entity.get("chunk_id"),
                        "parent_document_id": hit.entity.get("parent_document_id"),
                        "text": hit.entity.get("text"),
                        "source_file": hit.entity.get("source_file"),
                        "metadata": {
                            "chunk_size_tokens": hit.entity.get("chunk_size_tokens"),
                            "estimated_tokens": hit.entity.get("estimated_tokens")
                        },
                        "score": hit.distance,  # Cosine similarity score
                        "milvus_id": hit.id
                    }
                    retrieved_chunks.append(chunk)
            
            logger.info(f"Vector search returned {len(retrieved_chunks)} results")
            return retrieved_chunks
            
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
            # Query parent collection
            parent_map = {}
            
            for parent_id in parent_ids:
                # Search by document_id field
                expr = f'document_id == "{parent_id}"'
                results = self.parent_collection.query(
                    expr=expr,
                    output_fields=["document_id", "source_file", "total_chunks", 
                                 "chunk_strategy", "chunking_method", "total_pages"]
                )
                
                if results:
                    parent_map[parent_id] = results[0]
            
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
            min_chunk_id = max(0, current_chunk_id - window_size)
            max_chunk_id = current_chunk_id + window_size
            
            expr = f'parent_document_id == "{parent_id}" && chunk_id >= {min_chunk_id} && chunk_id <= {max_chunk_id}'
            
            results = self.chunk_collection.query(
                expr=expr,
                output_fields=["chunk_id", "parent_document_id", "text", "source_file"]
            )
            
            # Sort by chunk_id
            results.sort(key=lambda x: x["chunk_id"])
            
            return results
            
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
        
        if not self.milvus_available or self.chunk_collection is None:
            logger.warning("Milvus not available for retrieval")
            return {
                "query": query,
                "results": [],
                "metadata": {
                    "total_results": 0,
                    "milvus_available": False,
                    "error": "Milvus collections not available"
                }
            }
        
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
                "metadata": {
                    "total_results": 0,
                    "milvus_available": True,
                    "error": "No matching documents found"
                }
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
                "rerank_metrics": rerank_metrics,
                "milvus_available": True,
                "model_key": self.model_key
            }
        }
        
        logger.info(f"Retrieved {len(reranked_results)} results from {len(parent_ids)} documents")
        return response
    
    def semantic_search_with_filters(self, query: str, 
                                    parent_document_id: str = None,
                                    top_k: int = 10) -> Dict:
        """
        Semantic search with optional filters.
        
        Args:
            query: Query string
            parent_document_id: Filter by specific parent document
            top_k: Number of results
            
        Returns:
            Filtered search results
        """
        query_embedding = self.encode_query(query)
        
        # Build filter expression
        filter_expr = None
        if parent_document_id:
            filter_expr = f'parent_document_id == "{parent_document_id}"'
        
        try:
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # Search with filter
            if filter_expr:
                results = self.chunk_collection.search(
                    data=[query_embedding.tolist()],
                    anns_field="embedding",
                    param=search_params,
                    limit=top_k,
                    expr=filter_expr,
                    output_fields=["chunk_id", "parent_document_id", "text", "source_file"]
                )
            else:
                results = self.chunk_collection.search(
                    data=[query_embedding.tolist()],
                    anns_field="embedding",
                    param=search_params,
                    limit=top_k,
                    output_fields=["chunk_id", "parent_document_id", "text", "source_file"]
                )
            
            # Parse results
            retrieved = []
            for hits in results:
                for hit in hits:
                    retrieved.append({
                        "chunk_id": hit.entity.get("chunk_id"),
                        "parent_document_id": hit.entity.get("parent_document_id"),
                        "text": hit.entity.get("text"),
                        "source_file": hit.entity.get("source_file"),
                        "score": hit.distance
                    })
            
            return {
                "query": query,
                "results": retrieved,
                "filters": {"parent_document_id": parent_document_id},
                "metadata": {"total_results": len(retrieved)}
            }
            
        except Exception as e:
            logger.error(f"Error during filtered search: {str(e)}")
            return {"query": query, "results": [], "error": str(e)}


def main():
    """Main function to demonstrate Milvus retrieval pipeline."""
    # Define paths
    db_path = "/media/data/codes/reshma/lma_maj_pro/partc/milvus_data.db"
    
    # Initialize pipeline
    try:
        pipeline = MilvusRetrievalPipeline(db_path)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        return
    
    # Example queries
    test_queries = [
        "What are the main causes of climate change?",
        "How does deforestation affect biodiversity?",
        "What are sustainable development practices?",
        "Explain the greenhouse effect"
    ]
    
    logger.info("="*60)
    logger.info("Testing Milvus Retrieval Pipeline")
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
            score = top_result.get('rerank_score', top_result.get('score'))
            logger.info(f"\nTop Result:")
            logger.info(f"  Score: {score:.4f}")
            logger.info(f"  Document: {top_result.get('source_file', 'Unknown')}")
            logger.info(f"  Text preview: {top_result['text'][:200]}...")
        
        logger.info("\n")


if __name__ == "__main__":
    main()
    
