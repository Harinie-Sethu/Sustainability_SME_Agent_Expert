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
project_root = Path(__file__).parent.parent

from embedding_generator import EmbeddingGenerator
from vector_indexer_milvus import MilvusVectorIndexer as VectorIndexer
from milvus_retrieval_pipeline import MilvusRetrievalPipeline  # CHANGED

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function for Part C."""
    
    # Define paths - process files from BOTH old and new chunked_data directories
    chunked_data_dirs = [
        project_root / "partb" / "chunked_data",
        project_root / "partb" / "output" / "chunked_data"
    ]
    output_dir = project_root / "partc"
    embeddings_dir = output_dir / "embeddings"
    db_path = output_dir / "milvus_lite.db"
    if not db_path.exists():
        db_path = output_dir / "milvus_data.db"
    
    logger.info("="*70)
    logger.info("PART C: EMBEDDING & INDEXING PIPELINE")
    logger.info("="*70)
    
    # Step 1: Generate Embeddings
    logger.info("\n" + "="*70)
    logger.info("STEP 1: GENERATING EMBEDDINGS")
    logger.info("="*70 + "\n")
    
    # Collect all chunked files from both directories
    all_chunked_files = []
    for chunked_dir in chunked_data_dirs:
        if chunked_dir.exists():
            files = list(chunked_dir.glob("*_chunked.json"))
            logger.info(f"Found {len(files)} chunked files in {chunked_dir}")
            all_chunked_files.extend(files)
        else:
            logger.warning(f"Directory does not exist: {chunked_dir}")
    
    if not all_chunked_files:
        logger.error("No chunked files found in any directory!")
        return 1
    
    logger.info(f"Total chunked files to process: {len(all_chunked_files)}\n")
    
    try:
        # Process files from each directory separately (or use first dir as base)
        # We'll process all files regardless of which directory they're in
        results = []
        successful_processing = 0
        total_chunks = 0
        
        # Use the first directory as the base for the generator
        base_chunked_dir = chunked_data_dirs[0]
        generator = EmbeddingGenerator(str(base_chunked_dir), str(output_dir))
        
        # Process each file individually
        logger.info(f"Processing {len(all_chunked_files)} chunked files...\n")
        for idx, chunked_file in enumerate(all_chunked_files, 1):
            logger.info(f"[{idx}/{len(all_chunked_files)}] Processing: {chunked_file.name}")
            result = generator.process_chunked_file(
                chunked_file,
                chunk_strategy="medium",
                chunking_method="content_aware"
            )
            results.append(result)
            
            if result.get("success", False):
                successful_processing += 1
                total_chunks += result.get("num_chunks", 0)
                logger.info(f"  ✓ Success: {result.get('num_chunks', 0)} chunks embedded")
            else:
                logger.error(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
        
        embedding_summary = {
            "embedding_generation_summary": {
                "successful_processing": successful_processing,
                "failed_processing": len(all_chunked_files) - successful_processing,
                "total_chunks_embedded": total_chunks,
                "total_files": len(all_chunked_files)
            },
            "file_results": results
        }
        
        logger.info("\nEmbedding generation completed successfully!")
        logger.info(f"Total files processed: {embedding_summary['embedding_generation_summary']['successful_processing']}")
        logger.info(f"Total chunks embedded: {embedding_summary['embedding_generation_summary']['total_chunks_embedded']}")
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        return 1
    
    # Step 2: Index into Milvus
    logger.info("\n" + "="*70)
    logger.info("STEP 2: INDEXING INTO MILVUS")
    logger.info("="*70 + "\n")
    
    try:
        indexer = VectorIndexer(str(embeddings_dir), str(db_path))
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
        logger.error("Make sure pymilvus is installed: pip install pymilvus milvus")
        return 1
    
    # Step 3: Test Milvus Retrieval Pipeline with Reranking
    logger.info("\n" + "="*70)
    logger.info("STEP 3: TESTING MILVUS RETRIEVAL PIPELINE WITH RERANKING")
    logger.info("="*70 + "\n")
    
    try:
        pipeline = MilvusRetrievalPipeline(db_path)  # CHANGED
        
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
                rerank=True  # Reranking enabled!
            )
            
            logger.info(f"  Retrieved {len(results['results'])} results")
            logger.info(f"  Reranking applied: {results['metadata']['reranking_applied']}")
            
            if results['results']:
                top_score = results['results'][0].get('rerank_score', results['results'][0].get('score', 0))
                logger.info(f"  Top result score: {top_score:.4f}")
                logger.info(f"  Source: {results['results'][0].get('source_file', 'Unknown')}")
        
        logger.info("\nMilvus retrieval pipeline test completed successfully!")
        
    except Exception as e:
        logger.error(f"Retrieval test failed: {str(e)}")
        logger.warning("Retrieval pipeline requires reranker model to be downloaded")
        return 1
    
    # Final Summary
    logger.info("\n" + "="*70)
    logger.info("PART C PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*70 + "\n")
    logger.info(f"Milvus database created at: {db_path}")
    logger.info(f"Embeddings stored in: {embeddings_dir}")
    logger.info("Reranking with BGE Reranker:  Enabled")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
