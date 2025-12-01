#!/usr/bin/env python3
"""
Index all existing documents into Milvus
Run this to populate Milvus with all documents from data_json/ and embeddings/
"""

# CRITICAL: Set protobuf implementation BEFORE any other imports
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
from document_indexer import index_all_existing_documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Index all existing documents."""
    project_root = Path(__file__).parent.parent
    
    logger.info("="*80)
    logger.info("INDEXING ALL EXISTING DOCUMENTS INTO MILVUS")
    logger.info("="*80)
    logger.info("")
    
    result = index_all_existing_documents(project_root, force_reindex=False)
    
    if result.get('success', False):
        logger.info("")
        logger.info("="*80)
        logger.info("✓ INDEXING COMPLETE!")
        logger.info("="*80)
        logger.info(f"Total files: {result.get('total_files', 0)}")
        logger.info(f"Successful: {result.get('successful', 0)}")
        logger.info(f"Failed: {result.get('failed', 0)}")
        logger.info("")
        logger.info("Collections are now ready for retrieval!")
        return 0
    else:
        logger.error(f"Indexing failed: {result.get('error', 'Unknown error')}")
        return 1

if __name__ == "__main__":
    exit(main())

