#!/usr/bin/env python3
"""
Main runner script for Part C - Embedding & Indexing Pipeline
Orchestrates embedding generation, indexing, and retrieval setup.
"""

import sys
import logging
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from embedding_generator import EmbeddingGenerator
from vector_indexer import MilvusVectorIndexer as VectorIndexer
from retrieval_pipeline import RetrievalPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function for Part C."""
    
    # Define paths
    chunked_data_dir = "/media/data/codes/reshma/lma_maj_pro/partb/chunked_data"
    output_dir = "/media/data/codes/reshma/lma_maj_pro/partc"
    embeddings_dir = f"{output_dir}/embeddings"
    db_path = f"{output_dir}/milvus_data.db"
    
    # Elasticsearch configuration (for retrieval pipeline only)
    es_host = "localhost"
    es_port = 9200
    
    logger.info("="*70)
    logger.info("PART C: EMBEDDING & INDEXING PIPELINE")
    logger.info("="*70)
    
    '''# Step 1: Generate Embeddings
    logger.info("\n" + "="*70)
    logger.info("STEP 1: GENERATING EMBEDDINGS")
    logger.info("="*70 + "\n")
    
    try:
        generator = EmbeddingGenerator(chunked_data_dir, output_dir)
        
        # Generate embeddings for medium chunks with content-aware chunking
        # This is the recommended configuration
        embedding_summary = generator.process_all_files(
            chunk_strategy="medium",
            chunking_method="content_aware"
        )
        
        logger.info("\nEmbedding generation completed successfully!")
        logger.info(f"Total files processed: {embedding_summary['embedding_generation_summary']['successful_processing']}")
        logger.info(f"Total chunks embedded: {embedding_summary['embedding_generation_summary']['total_chunks_embedded']}")
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        return 1'''
    
    # Step 2: Index into Milvus
    logger.info("\n" + "="*70)
    logger.info("STEP 2: INDEXING INTO MILVUS")
    logger.info("="*70 + "\n")
    
    try:
        # Correct initialization - only embeddings_dir and db_path
        indexer = VectorIndexer(embeddings_dir, db_path)
        indexing_summary = indexer.index_all_documents()
        
        logger.info("\nIndexing completed successfully!")
        logger.info(f"Total documents indexed: {indexing_summary['indexing_summary']['successful_indexing']}")
        logger.info(f"Total chunks indexed: {indexing_summary['indexing_summary']['total_chunks_indexed']}")
        
        # Get index statistics
        stats = indexer.get_index_stats()
        logger.info(f"\nIndex Statistics:")
        logger.info(f"  Parent documents: {stats.get('parent_documents', 0)}")
        logger.info(f"  Total chunks: {stats.get('total_chunks', 0)}")
        logger.info(f"  Avg chunks/document: {stats.get('avg_chunks_per_document', 0):.2f}")
        
    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}")
        logger.error("Make sure pymilvus is installed: pip install pymilvus")
        return 1
    
    # Step 3: Test Retrieval Pipeline (if using Elasticsearch)
    logger.info("\n" + "="*70)
    logger.info("STEP 3: TESTING RETRIEVAL PIPELINE")
    logger.info("="*70 + "\n")
    
    try:
        pipeline = RetrievalPipeline(es_host, es_port)
        
        # Test queries
        test_queries = [
            "What are the causes of climate change?",
            "How does pollution affect the environment?"
        ]
        
        for query in test_queries:
            logger.info(f"\nTesting query: '{query}'")
            results = pipeline.hierarchical_retrieve(
                query,
                top_k=3,
                expand_context=True,
                rerank=True
            )
            
            logger.info(f"  Retrieved {len(results['results'])} results")
            if results['results']:
                top_score = results['results'][0].get('rerank_score', results['results'][0].get('score', 0))
                logger.info(f"  Top result score: {top_score:.4f}")
        
        logger.info("\nRetrieval pipeline test completed successfully!")
        
    except Exception as e:
        logger.warning(f"Retrieval test skipped: {str(e)}")
        logger.warning("Note: Retrieval pipeline requires Elasticsearch to be running")
        logger.info("Milvus indexing completed successfully. You can set up retrieval later.")
    
    # Final Summary
    logger.info("\n" + "="*70)
    logger.info("PART C PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*70 + "\n")
    logger.info(f"Milvus database created at: {db_path}")
    logger.info(f"Embeddings stored in: {embeddings_dir}")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
