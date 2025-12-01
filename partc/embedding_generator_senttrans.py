#!/usr/bin/env python3
"""
Embedding Generation Pipeline for Environment/Sustainability Documents
Generates embeddings using multiple models for comparison and stores them with metadata.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for chunked documents using multiple models."""
    
    def __init__(self, chunked_data_dir: str, output_dir: str):
        """
        Initialize the embedding generator.
        
        Args:
            chunked_data_dir: Directory containing chunked JSON files
            output_dir: Directory to save embeddings
        """
        self.chunked_data_dir = Path(chunked_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different models
        self.embeddings_dir = self.output_dir / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)
        
        # Check for GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.models = self._initialize_models()
        
    def _initialize_models(self) -> Dict[str, SentenceTransformer]:
        """Initialize embedding models."""
        models = {}
        
        try:
            # Primary model: all-mpnet-base-v2 (general purpose, high quality)
            logger.info("Loading primary model: all-mpnet-base-v2")
            models['all-mpnet-base-v2'] = SentenceTransformer('all-mpnet-base-v2', device=self.device)
            
            # Domain-specific model: allenai-specter (scientific documents)
            logger.info("Loading domain-specific model: allenai-specter")
            models['allenai-specter'] = SentenceTransformer('allenai-specter', device=self.device)
            
            logger.info(f"Successfully loaded {len(models)} embedding models")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
        
        return models
    
    def generate_embeddings(self, texts: List[str], model_name: str, batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            model_name: Name of the model to use
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        try:
            # Generate embeddings in batches
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalization for cosine similarity
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def process_chunked_file(self, chunked_file: Path, chunk_strategy: str = "medium", 
                            chunking_method: str = "content_aware") -> Dict:
        """
        Process a single chunked file and generate embeddings.
        
        Args:
            chunked_file: Path to chunked JSON file
            chunk_strategy: Chunk size strategy ('small', 'medium', 'large')
            chunking_method: Chunking method ('fixed_size', 'content_aware', 'recursive_character')
            
        Returns:
            Processing results dictionary
        """
        logger.info(f"Processing embeddings for: {chunked_file.name}")
        
        try:
            # Load chunked data
            with open(chunked_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data.get("chunking_success", False):
                return {
                    "file": chunked_file.name,
                    "success": False,
                    "error": "Chunking failed"
                }
            
            # Extract chunks for specified strategy and method
            chunks_data = data.get("chunks", {}).get(chunk_strategy, {}).get(chunking_method, [])
            
            if not chunks_data:
                return {
                    "file": chunked_file.name,
                    "success": False,
                    "error": f"No chunks found for {chunk_strategy}/{chunking_method}"
                }
            
            # Extract texts from chunks
            chunk_texts = [chunk["text"] for chunk in chunks_data]
            
            # Generate embeddings for each model
            embeddings_by_model = {}
            
            for model_name in self.models.keys():
                logger.info(f"Generating embeddings with {model_name} for {chunked_file.name}")
                
                embeddings = self.generate_embeddings(chunk_texts, model_name)
                
                # Store embeddings with chunk metadata
                embeddings_by_model[model_name] = {
                    "embeddings": embeddings.tolist(),  # Convert to list for JSON serialization
                    "embedding_dim": embeddings.shape[1],
                    "num_chunks": len(chunk_texts)
                }
            
            # Create parent-child relationships
            # Parent = document level, Children = chunks
            parent_metadata = {
                "document_id": chunked_file.stem.replace('_chunked', ''),
                "source_file": data.get("source_file", ""),
                "total_chunks": len(chunks_data),
                "chunk_strategy": chunk_strategy,
                "chunking_method": chunking_method,
                "original_metadata": data.get("original_metadata", {}),
                "filtering_stats": data.get("filtering_stats", {})
            }
            
            # Create chunk-level metadata
            chunks_with_embeddings = []
            for idx, chunk in enumerate(chunks_data):
                chunk_entry = {
                    "chunk_id": chunk["chunk_id"],
                    "parent_document_id": parent_metadata["document_id"],
                    "text": chunk["text"],
                    "metadata": {
                        "chunk_size_tokens": chunk.get("chunk_size_tokens", 0),
                        "estimated_tokens": chunk.get("estimated_tokens", 0),
                        "text_length": chunk.get("text_length", 0),
                        "start_char": chunk.get("start_char", 0),
                        "end_char": chunk.get("end_char", 0)
                    },
                    "embeddings": {
                        model_name: embeddings_by_model[model_name]["embeddings"][idx]
                        for model_name in self.models.keys()
                    }
                }
                chunks_with_embeddings.append(chunk_entry)
            
            # Create output structure
            output_data = {
                "parent_metadata": parent_metadata,
                "chunks": chunks_with_embeddings,
                "embedding_models": {
                    model_name: {
                        "embedding_dim": embeddings_by_model[model_name]["embedding_dim"],
                        "model_name": model_name
                    }
                    for model_name in self.models.keys()
                },
                "generation_timestamp": datetime.now().isoformat(),
                "embedding_generation_success": True
            }
            
            # Save embeddings
            output_filename = f"{parent_metadata['document_id']}_{chunk_strategy}_{chunking_method}_embeddings.json"
            output_path = self.embeddings_dir / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Generated embeddings for {chunked_file.name}: {len(chunks_with_embeddings)} chunks")
            
            return {
                "file": chunked_file.name,
                "success": True,
                "output_file": output_filename,
                "num_chunks": len(chunks_with_embeddings),
                "models_used": list(self.models.keys()),
                "chunk_strategy": chunk_strategy,
                "chunking_method": chunking_method
            }
            
        except Exception as e:
            logger.error(f"Error processing {chunked_file.name}: {str(e)}")
            return {
                "file": chunked_file.name,
                "success": False,
                "error": str(e)
            }
    
    def process_all_files(self, chunk_strategy: str = "medium", 
                         chunking_method: str = "content_aware") -> Dict:
        """
        Process all chunked files and generate embeddings.
        
        Args:
            chunk_strategy: Chunk size strategy
            chunking_method: Chunking method
            
        Returns:
            Summary of processing results
        """
        chunked_files = list(self.chunked_data_dir.glob("*_chunked.json"))
        
        if not chunked_files:
            logger.warning(f"No chunked files found in {self.chunked_data_dir}")
            return {"error": "No files to process"}
        
        logger.info(f"Found {len(chunked_files)} chunked files to process")
        logger.info(f"Using chunk strategy: {chunk_strategy}, method: {chunking_method}")
        
        results = []
        successful_processing = 0
        total_chunks = 0
        
        for chunked_file in tqdm(chunked_files, desc="Generating embeddings"):
            result = self.process_chunked_file(chunked_file, chunk_strategy, chunking_method)
            results.append(result)
            
            if result.get("success", False):
                successful_processing += 1
                total_chunks += result.get("num_chunks", 0)
        
        # Generate summary
        summary = {
            "embedding_generation_summary": {
                "total_files": len(chunked_files),
                "successful_processing": successful_processing,
                "failed_processing": len(chunked_files) - successful_processing,
                "total_chunks_embedded": total_chunks,
                "models_used": list(self.models.keys()),
                "chunk_strategy": chunk_strategy,
                "chunking_method": chunking_method,
                "device_used": self.device,
                "generation_timestamp": datetime.now().isoformat()
            },
            "file_results": results
        }
        
        # Save summary
        summary_path = self.output_dir / f"embedding_generation_summary_{chunk_strategy}_{chunking_method}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Embedding generation complete: {successful_processing}/{len(chunked_files)} files processed")
        logger.info(f"Total chunks embedded: {total_chunks}")
        
        return summary


def main():
    """Main function to run embedding generation."""
    # Define paths
    chunked_data_dir = "/media/data/codes/reshma/lma_maj_pro/partb/chunked_data"
    output_dir = "/media/data/codes/reshma/lma_maj_pro/partc"
    
    # Create generator instance
    generator = EmbeddingGenerator(chunked_data_dir, output_dir)
    
    # Process all files with different configurations
    # You can run this for different chunk strategies and methods
    configurations = [
        ("medium", "content_aware"),  # Recommended for most use cases
        # ("small", "content_aware"),   # For fine-grained retrieval
        # ("large", "content_aware"),   # For broader context
    ]
    
    for chunk_strategy, chunking_method in configurations:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing configuration: {chunk_strategy} / {chunking_method}")
        logger.info(f"{'='*60}\n")
        
        summary = generator.process_all_files(chunk_strategy, chunking_method)
    
    logger.info("Embedding generation pipeline completed!")
    logger.info(f"Check the output directory: {output_dir}")


if __name__ == "__main__":
    main()
