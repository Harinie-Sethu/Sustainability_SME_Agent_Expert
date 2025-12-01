#!/usr/bin/env python3
"""
Document Indexer - Process and index new documents into Milvus
Handles: chunking -> embedding generation -> Milvus indexing
"""

# CRITICAL: Set protobuf implementation BEFORE any other imports
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_and_index_document(
    json_file_path: Path,
    project_root: Path,
    chunk_strategy: str = "medium",
    chunking_method: str = "content_aware",
    chunked_file_path: Optional[Path] = None
) -> Dict:
    """
    Process a single JSON document: generate embeddings -> index into Milvus.
    Assumes the document has already been chunked by batch_pipeline.
    
    Args:
        json_file_path: Path to JSON file in data_json/ (e.g., sus.json)
        project_root: Project root directory
        chunk_strategy: Chunking strategy (small/medium/large)
        chunking_method: Chunking method (content_aware/sentence/token)
    
    Returns:
        Processing results dictionary
    """
    logger.info(f"Processing and indexing document: {json_file_path.name}")
    
    try:
        chunked_file = None
        chunked_data_dir = None
        
        if chunked_file_path:
            candidate = Path(chunked_file_path)
            logger.info(f"Using provided chunked file path: {candidate}")
            if candidate.exists():
                chunked_file = candidate
                chunked_data_dir = candidate.parent
            else:
                logger.warning(f"Provided chunked file does not exist yet: {candidate}")
        
        if chunked_file is None:
            # Step 1: Find the chunked file (created by batch_pipeline)
            # Support both legacy chunked directory and new partb/output/chunked_data
            chunked_dirs = [
                project_root / "partb" / "chunked_data",
                project_root / "partb" / "output" / "chunked_data"
            ]
            chunked_filename = f"{json_file_path.stem}_chunked.json"
            
            for directory in chunked_dirs:
                candidate = directory / chunked_filename
                logger.info(f"Looking for chunked file: {candidate}")
                if candidate.exists():
                    chunked_file = candidate
                    chunked_data_dir = directory
                    break
            
            if chunked_file is None:
                logger.warning("Exact chunked file not found. Performing fuzzy search...")
                similar_files = []
                for directory in chunked_dirs:
                    similar_files.extend(directory.glob(f"*{json_file_path.stem}*chunked.json"))
                    if similar_files:
                        chunked_file = similar_files[0]
                        chunked_data_dir = chunked_file.parent
                        logger.info(f"Found similar chunked file: {chunked_file}")
                        break
            
            if chunked_file is None:
                # List all available chunked files for debugging
                available_files = {}
                for directory in chunked_dirs:
                    files = [f.name for f in directory.glob("*chunked.json")]
                    available_files[str(directory)] = files
                
                logger.error(f"Chunked file not found. Expected: {chunked_filename}")
                logger.error(f"Available chunked files: {available_files}")
                return {
                    "success": False,
                    "error": f"Chunked file not found: {chunked_filename}. Available files: {available_files}. Make sure document was processed through batch pipeline."
                }
        
        logger.info(f"✓ Found chunked file: {chunked_file.name}")
        
        # Step 2: Generate embeddings
        logger.info("Step 2: Generating embeddings...")
        sys.path.insert(0, str(project_root / "partc"))
        from embedding_generator import EmbeddingGenerator
        
        output_dir = project_root / "partc"
        embeddings_dir = output_dir / "embeddings"
        embeddings_dir.mkdir(exist_ok=True, parents=True)
        
        generator = EmbeddingGenerator(
            str(chunked_data_dir),
            str(output_dir),
            models_to_use=['baseline', 'scientific']
        )
        
        # Process the chunked file
        embedding_result = generator.process_chunked_file(
            chunked_file,
            chunk_strategy=chunk_strategy,
            chunking_method=chunking_method
        )
        
        if not embedding_result.get('success', False):
            return {
                "success": False,
                "error": f"Embedding generation failed: {embedding_result.get('error', 'Unknown error')}"
            }
        
        embedding_file = embedding_result.get('output_file')
        if not embedding_file:
            return {
                "success": False,
                "error": "Embedding file path not returned"
            }
        
        embedding_path = Path(embedding_file)
        if not embedding_path.exists():
            # Try to find it in embeddings_dir
            embedding_path = embeddings_dir / embedding_path.name
            if not embedding_path.exists():
                return {
                    "success": False,
                    "error": f"Embedding file was not created: {embedding_file}"
                }
        
        logger.info(f"✓ Generated embeddings: {embedding_path.name}")
        
        # Step 3: Index into Milvus
        logger.info("Step 3: Indexing into Milvus...")
        from vector_indexer_milvus import MilvusVectorIndexer
        
        # Use milvus_lite.db (same as retrieval pipeline) for consistency
        db_path = project_root / "partc" / "milvus_lite.db"
        # Fallback to milvus_data.db if milvus_lite.db doesn't exist
        if not db_path.exists():
            db_path = project_root / "partc" / "milvus_data.db"
        indexer = MilvusVectorIndexer(str(embeddings_dir), str(db_path))
        
        # Check if collections exist, create if not
        from pymilvus import utility
        available_collections = utility.list_collections()
        
        if not available_collections:
            logger.info("No collections found, creating...")
            # Get embedding info to determine dimensions
            with open(embedding_path, 'r', encoding='utf-8') as f:
                emb_data = json.load(f)
            embedding_models_info = emb_data.get("embedding_models", {})
            if embedding_models_info:
                first_model = list(embedding_models_info.keys())[0]
                embedding_dim = embedding_models_info[first_model]["embedding_dim"]
                available_models = list(embedding_models_info.keys())
                indexer.create_collections(embedding_dim, available_models)
                logger.info(f"✓ Created collections for models: {available_models}")
        
        # Index the document
        logger.info(f"Indexing document into Milvus database: {db_path}")
        indexing_result = indexer.index_document(embedding_path)
        
        if not indexing_result.get('success', False):
            error_msg = indexing_result.get('error', 'Unknown error')
            logger.error(f"✗ Indexing failed: {error_msg}")
            return {
                "success": False,
                "error": f"Indexing failed: {error_msg}"
            }
        
        # Flush collections to ensure data is persisted
        try:
            logger.info("Flushing collections to persist data...")
            for model_name, collection in indexer.chunk_collections.items():
                collection.flush()
                collection.load()
                logger.info(f"  ✓ Flushed and loaded collection for {model_name}")
            if indexer.parent_collection:
                indexer.parent_collection.flush()
                logger.info("  ✓ Flushed parent collection")
        except Exception as e:
            logger.warning(f"Could not flush collections: {e}")
            # Don't fail indexing if flush fails
        
        chunks_per_model = indexing_result.get('chunks_indexed_per_model', {})
        total_chunks = indexing_result.get('total_chunks', 0)
        
        logger.info(f"✓ Indexed into Milvus: {total_chunks} chunks")
        for model, count in chunks_per_model.items():
            logger.info(f"  - {model}: {count} chunks")
        
        return {
            "success": True,
            "json_file": json_file_path.name,
            "chunked_file": chunked_file.name,
            "embedding_file": embedding_path.name,
            "chunks_indexed": total_chunks,
            "chunks_per_model": chunks_per_model
        }
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }


def index_all_existing_documents(project_root: Path, force_reindex: bool = False) -> Dict:
    """
    Index all existing documents from partc/embeddings/ into Milvus.
    
    Args:
        project_root: Project root directory
        force_reindex: If True, reindex even if already indexed
    
    Returns:
        Indexing results
    """
    logger.info("Indexing all existing documents...")
    
    # Check if embeddings exist
    embeddings_dir = project_root / "partc" / "embeddings"
    embedding_files = list(embeddings_dir.glob("*_embeddings.json")) if embeddings_dir.exists() else []
    
    if not embedding_files:
        logger.warning("No embedding files found. Need to generate embeddings first.")
        logger.info("Run: python partc/run_partc_milvus.py to generate embeddings and index")
        return {
            "success": False,
            "error": "No embedding files found. Generate embeddings first."
        }
    
    logger.info(f"Found {len(embedding_files)} existing embedding files, indexing into Milvus...")
    
    # Index existing embeddings
    # Use milvus_lite.db (same as retrieval pipeline) for consistency
    db_path = project_root / "partc" / "milvus_lite.db"
    # Fallback to milvus_data.db if milvus_lite.db doesn't exist
    if not db_path.exists():
        db_path = project_root / "partc" / "milvus_data.db"
    logger.info(f"Using Milvus database: {db_path}")
    
    try:
        from partc.vector_indexer_milvus import MilvusVectorIndexer
        indexer = MilvusVectorIndexer(str(embeddings_dir), str(db_path))
        
        # Get embedding info
        with open(embedding_files[0], 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
        embedding_models_info = sample_data.get("embedding_models", {})
        
        if not embedding_models_info:
            return {
                "success": False,
                "error": "No embedding models found in data"
            }
        
        first_model = list(embedding_models_info.keys())[0]
        embedding_dim = embedding_models_info[first_model]["embedding_dim"]
        available_models = list(embedding_models_info.keys())
        
        # Create collections if needed
        from pymilvus import utility
        available_collections = utility.list_collections()
        
        if not available_collections or force_reindex:
            if force_reindex and available_collections:
                logger.info("Dropping existing collections for reindexing...")
                indexer.drop_collections_if_exist()
            
            logger.info("Creating Milvus collections...")
            indexer.create_collections(embedding_dim, available_models)
            logger.info(f"✓ Created collections for models: {available_models}")
        else:
            logger.info(f"Collections already exist: {available_collections}")
        
        # Index all documents
        logger.info(f"Indexing {len(embedding_files)} documents...")
        summary = indexer.index_all_documents()
        
        # Get stats
        stats = indexer.get_index_stats()
        
        return {
            "success": True,
            "indexing_summary": summary.get('indexing_summary', {}),
            "stats": stats,
            "total_files": len(embedding_files)
        }
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}




