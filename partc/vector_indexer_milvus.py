#!/usr/bin/env python3
"""
Vector Indexing Pipeline using Milvus for Environment/Sustainability Documents
Milvus is a high-performance vector database designed for similarity search.
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

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

try:
    from pymilvus import (
        connections,
        utility,
        FieldSchema, CollectionSchema, DataType,
        Collection,
    )
except ImportError as e:
    import sys
    print(f"ERROR: pymilvus not installed: {e}")
    print("Please install with: pip install pymilvus milvus")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vector_indexing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MilvusVectorIndexer:
    """Index embeddings into Milvus with parent-child relationships."""
    
    def __init__(self, embeddings_dir: str, db_path: str = None):
        """
        Initialize the Milvus indexer.
        
        Args:
            embeddings_dir: Directory containing embedding files
            db_path: Path to store Milvus data (uses Milvus Lite - local file-based)
        """
        self.embeddings_dir = Path(embeddings_dir)
        
        if db_path is None:
            db_path = str(self.embeddings_dir.parent / "milvus_data.db")
        
        self.db_path = db_path
        
        # Connect to Milvus (try server first, then use Milvus Lite)
        try:
            # First try localhost:19530 (standard Milvus port)
            connections.connect(alias="default", host="localhost", port="19530", timeout=10)
            logger.info("Connected to Milvus server on localhost:19530")
        except Exception as e1:
            logger.warning(f"Could not connect to Milvus server on localhost:19530: {e1}")
            # Try starting Milvus Lite as fallback
            try:
                logger.info("Attempting to start Milvus Lite...")
                import time
                
                # Try different import paths for Milvus Lite
                default_server = None
                try:
                    from milvus_lite import default_server
                except ImportError:
                    try:
                        from milvus import default_server
                    except ImportError:
                        try:
                            from pymilvus.server import default_server
                        except ImportError:
                            # Last resort: try URI-based connection (pymilvus 2.6+)
                            logger.info("Trying URI-based connection (pymilvus 2.6+)...")
                            db_dir = Path(db_path).parent if Path(db_path).is_file() else Path(db_path)
                            db_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                            uri_path = str(db_dir / "milvus_lite.db")
                            try:
                                # Try different URI formats
                                try:
                                    connections.connect(alias="default", uri=uri_path)
                                    logger.info(f"Connected to Milvus Lite via URI: {uri_path}")
                                except Exception:
                                    # Try with file:// prefix
                                    connections.connect(alias="default", uri=f"file://{uri_path}")
                                    logger.info(f"Connected to Milvus Lite via URI (file://): {uri_path}")
                                # Skip server startup if URI connection works
                                default_server = None
                            except Exception as uri_error:
                                raise ImportError(
                                    f"Milvus Lite not available. URI connection also failed: {uri_error}. "
                                    "Install with: pip install 'pymilvus[milvus_lite]' or pip install milvus-lite"
                                )
                
                # If we have default_server, start it
                if default_server is not None:
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
        
        # Collection names
        self.parent_collection_name = "environment_documents"
        self.chunk_collection_name = "environment_chunks"
        
        # Will store collections
        self.parent_collection = None
        self.chunk_collections = {}
        self.available_models = []
        
        self._load_existing_collections()

    def _load_existing_collections(self):
        """Load existing collections if they already exist."""
        try:
            existing_collections = utility.list_collections()
        except Exception as e:
            logger.warning(f"Could not list existing Milvus collections: {e}")
            return
        
        if self.parent_collection_name in existing_collections:
            try:
                self.parent_collection = Collection(self.parent_collection_name)
                logger.info(f"✓ Loaded existing collection: {self.parent_collection_name}")
            except Exception as e:
                logger.warning(f"Failed to load parent collection {self.parent_collection_name}: {e}")
        
        for collection_name in existing_collections:
            if collection_name.startswith(self.chunk_collection_name):
                try:
                    collection = Collection(collection_name)
                    model_key = collection_name.replace(f"{self.chunk_collection_name}_", "")
                    self.chunk_collections[model_key] = collection
                    if model_key not in self.available_models:
                        self.available_models.append(model_key)
                    logger.info(f"✓ Loaded chunk collection: {collection_name}")
                except Exception as e:
                    logger.warning(f"Failed to load chunk collection {collection_name}: {e}")
    
    def drop_collections_if_exist(self):
        """Drop existing collections."""
        try:
            # Drop parent collection
            if utility.has_collection(self.parent_collection_name):
                utility.drop_collection(self.parent_collection_name)
                logger.info(f"Dropped existing collection: {self.parent_collection_name}")
            
            # Drop chunk collections
            for collection_name in utility.list_collections():
                if collection_name.startswith(self.chunk_collection_name):
                    utility.drop_collection(collection_name)
                    logger.info(f"Dropped existing collection: {collection_name}")
            
        except Exception as e:
            logger.warning(f"Error dropping collections: {str(e)}")
    
    def create_collections(self, embedding_dim: int, available_models: List[str]):
        """
        Create Milvus collections.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            available_models: List of available embedding model names
        """
        self.available_models = available_models
        
        # Drop existing collections
        self.drop_collections_if_exist()
        
        try:
            # Create parent collection schema
            # Note: Milvus 2.6+ requires at least one vector field, so we add a dummy vector field
            parent_fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
                FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="total_chunks", dtype=DataType.INT64),
                FieldSchema(name="chunk_strategy", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="chunking_method", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="total_pages", dtype=DataType.INT64),
                FieldSchema(name="dummy_vector", dtype=DataType.FLOAT_VECTOR, dim=1),  # Required for Milvus 2.6+
            ]
            
            parent_schema = CollectionSchema(
                fields=parent_fields,
                description="Parent documents metadata"
            )
            
            self.parent_collection = Collection(
                name=self.parent_collection_name,
                schema=parent_schema
            )
            logger.info(f"✓ Created collection: {self.parent_collection_name}")
            
            # Create chunk collections for each embedding model
            for model_name in available_models:
                collection_name = f"{self.chunk_collection_name}_{model_name.replace('-', '_').replace('/', '_')}"
                
                chunk_fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
                    FieldSchema(name="chunk_id", dtype=DataType.INT64),
                    FieldSchema(name="parent_document_id", dtype=DataType.VARCHAR, max_length=256),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=512),
                    FieldSchema(name="chunk_size_tokens", dtype=DataType.INT64),
                    FieldSchema(name="estimated_tokens", dtype=DataType.INT64),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
                ]
                
                chunk_schema = CollectionSchema(
                    fields=chunk_fields,
                    description=f"Document chunks with {model_name} embeddings"
                )
                
                collection = Collection(
                    name=collection_name,
                    schema=chunk_schema
                )
                
                # Create index on embedding field for faster search
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                collection.create_index(
                    field_name="embedding",
                    index_params=index_params
                )
                
                self.chunk_collections[model_name] = collection
                logger.info(f"✓ Created collection: {collection_name}")
            
            logger.info(f"Collections created for models: {available_models}")
            
        except Exception as e:
            logger.error(f"Error creating collections: {str(e)}")
            raise
    
    def index_document(self, embedding_file: Path) -> Dict:
        """
        Index a single document with its chunks.
        
        Args:
            embedding_file: Path to embedding JSON file
            
        Returns:
            Indexing results dictionary
        """
        logger.info(f"Indexing: {embedding_file.name}")
        
        try:
            # Load embedding data
            with open(embedding_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data.get("embedding_generation_success", False):
                return {
                    "file": embedding_file.name,
                    "success": False,
                    "error": "Embedding generation failed"
                }
            
            # Extract parent and chunk data
            parent_metadata = data.get("parent_metadata", {})
            chunks = data.get("chunks", [])
            
            # Index parent document
            parent_doc_id = parent_metadata["document_id"]
            
            try:
                parent_data = [
                    [parent_doc_id],  # id
                    [parent_doc_id],  # document_id
                    [parent_metadata.get("source_file", "")[:512]],  # source_file
                    [parent_metadata.get("total_chunks", 0)],  # total_chunks
                    [parent_metadata.get("chunk_strategy", "")[:64]],  # chunk_strategy
                    [parent_metadata.get("chunking_method", "")[:64]],  # chunking_method
                    [parent_metadata.get("original_metadata", {}).get("total_pages", 0)],  # total_pages
                    [[0.0]],  # dummy_vector (required for Milvus 2.6+, single dimension vector with value 0.0)
                ]
                
                self.parent_collection.insert(parent_data)
                logger.info(f"  ✓ Indexed parent: {parent_doc_id}")
                
            except Exception as e:
                logger.error(f"Error indexing parent document: {str(e)}")
                return {
                    "file": embedding_file.name,
                    "success": False,
                    "error": f"Parent indexing failed: {str(e)}"
                }
            
            # Index chunks for each embedding model
            success_count = {model: 0 for model in self.available_models}
            
            for model_name in self.available_models:
                collection = self.chunk_collections[model_name]
                
                # Prepare batch data
                ids = []
                chunk_ids = []
                parent_ids = []
                texts = []
                source_files = []
                chunk_size_tokens = []
                estimated_tokens = []
                embeddings = []
                
                for chunk in chunks:
                    chunk_id = f"{parent_doc_id}_chunk_{chunk['chunk_id']}"
                    
                    # Get embedding for this model
                    embedding = chunk["embeddings"].get(model_name, [])
                    if not embedding:
                        continue
                    
                    ids.append(chunk_id)
                    chunk_ids.append(chunk["chunk_id"])
                    parent_ids.append(parent_doc_id)
                    texts.append(chunk["text"][:65535])  # Truncate if too long
                    source_files.append(parent_metadata.get("source_file", "")[:512])
                    chunk_size_tokens.append(chunk["metadata"].get("chunk_size_tokens", 0))
                    estimated_tokens.append(chunk["metadata"].get("estimated_tokens", 0))
                    embeddings.append(embedding)
                
                # Insert batch
                if ids:
                    try:
                        data = [
                            ids,
                            chunk_ids,
                            parent_ids,
                            texts,
                            source_files,
                            chunk_size_tokens,
                            estimated_tokens,
                            embeddings
                        ]
                        
                        collection.insert(data)
                        success_count[model_name] = len(ids)
                        
                    except Exception as e:
                        logger.error(f"Error indexing chunks for {model_name}: {str(e)}")
            
            total_success = sum(success_count.values())
            logger.info(f"  ✓ Indexed {total_success} chunks across {len(self.available_models)} models")
            
            return {
                "file": embedding_file.name,
                "success": True,
                "parent_document_id": parent_doc_id,
                "chunks_indexed_per_model": success_count,
                "total_chunks": len(chunks)
            }
                
        except Exception as e:
            logger.error(f"Error processing {embedding_file.name}: {str(e)}")
            return {
                "file": embedding_file.name,
                "success": False,
                "error": str(e)
            }
    
    def index_all_documents(self) -> Dict:
        """
        Index all embedding files.
        
        Returns:
            Summary of indexing results
        """
        embedding_files = list(self.embeddings_dir.glob("*_embeddings.json"))
        
        if not embedding_files:
            logger.warning(f"No embedding files found in {self.embeddings_dir}")
            return {"error": "No files to index"}
        
        logger.info(f"Found {len(embedding_files)} embedding files to index")
        
        # Get embedding info from first file
        with open(embedding_files[0], 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
            embedding_models_info = sample_data.get("embedding_models", {})
            
            if not embedding_models_info:
                logger.error("No embedding models found in data!")
                return {"error": "No embedding models found"}
            
            first_model = list(embedding_models_info.keys())[0]
            embedding_dim = embedding_models_info[first_model]["embedding_dim"]
            available_models = list(embedding_models_info.keys())
            
            logger.info(f"Detected embedding dimension: {embedding_dim}")
            logger.info(f"Available models: {available_models}")
        
        # Create collections
        logger.info("\nCreating Milvus collections...")
        self.create_collections(embedding_dim, available_models)
        logger.info("✓ Collections created successfully\n")
        
        # Index all documents
        results = []
        successful_indexing = 0
        total_chunks_indexed = 0
        
        logger.info(f"Indexing {len(embedding_files)} documents...")
        for idx, embedding_file in enumerate(embedding_files, 1):
            logger.info(f"\n[{idx}/{len(embedding_files)}] Processing: {embedding_file.name}")
            result = self.index_document(embedding_file)
            results.append(result)
            
            if result.get("success", False):
                successful_indexing += 1
                chunks_per_model = result.get("chunks_indexed_per_model", {})
                if chunks_per_model:
                    total_chunks_indexed += list(chunks_per_model.values())[0]
        
        # Flush and load collections for search
        logger.info("\nFinalizing collections...")
        self.parent_collection.flush()
        for collection in self.chunk_collections.values():
            collection.flush()
            collection.load()
        
        # Generate summary
        summary = {
            "indexing_summary": {
                "total_files": len(embedding_files),
                "successful_indexing": successful_indexing,
                "failed_indexing": len(embedding_files) - successful_indexing,
                "total_chunks_indexed": total_chunks_indexed,
                "database_type": "Milvus",
                "database_path": self.db_path,
                "parent_collection": self.parent_collection_name,
                "chunk_collections": list(self.chunk_collections.keys()),
                "embedding_models": available_models,
                "indexing_timestamp": datetime.now().isoformat()
            },
            "file_results": results
        }
        
        # Save summary
        summary_path = self.embeddings_dir.parent / "indexing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Indexing complete: {successful_indexing}/{len(embedding_files)} files indexed")
        logger.info(f"Total chunks indexed: {total_chunks_indexed}")
        logger.info(f"Database location: {self.db_path}")
        logger.info(f"{'='*60}")
        
        return summary
    
    def get_index_stats(self) -> Dict:
        """Get statistics about indexed documents."""
        try:
            parent_count = self.parent_collection.num_entities
            
            chunk_counts = {}
            for model_name, collection in self.chunk_collections.items():
                chunk_counts[model_name] = collection.num_entities
            
            total_chunks = list(chunk_counts.values())[0] if chunk_counts else 0
            
            return {
                "database_type": "Milvus",
                "parent_documents": parent_count,
                "total_chunks": total_chunks,
                "chunks_per_model": chunk_counts,
                "avg_chunks_per_document": total_chunks / parent_count if parent_count > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {}


def main():
    """Main function to run vector indexing."""
    # Define paths
    embeddings_dir = "/media/data/codes/reshma/lma_maj_pro/partc/embeddings"
    db_path = "/media/data/codes/reshma/lma_maj_pro/partc/milvus_data.db"
    
    # Create indexer instance
    indexer = MilvusVectorIndexer(embeddings_dir, db_path)
    
    # Index all documents
    logger.info("Starting Milvus vector indexing pipeline...")
    summary = indexer.index_all_documents()
    
    # Get index statistics
    stats = indexer.get_index_stats()
    logger.info(f"\nIndex Statistics:")
    logger.info(f"  Database type: {stats.get('database_type', 'N/A')}")
    logger.info(f"  Parent documents: {stats.get('parent_documents', 0)}")
    logger.info(f"  Total chunks: {stats.get('total_chunks', 0)}")
    logger.info(f"  Avg chunks/document: {stats.get('avg_chunks_per_document', 0):.2f}")
    
    logger.info("\nMilvus vector indexing pipeline completed!")


if __name__ == "__main__":
    main()
