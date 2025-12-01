#!/usr/bin/env python3
"""
Comprehensive Batch Ingestion Pipeline for Environment/Sustainability Documents
Handles PDF extraction, cleaning, chunking, and automated processing of new documents.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib

# Import our processing modules
from pdf_text_extractor import PDFTextExtractor
from text_processor import EnvironmentTextProcessor
from text_chunker import TextChunker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BatchIngestionPipeline:
    """Comprehensive pipeline for processing PDF documents."""
    
    def __init__(self, 
                 dataset_dir: str,
                 data_for_finetuning_dir: str,
                 data_json_dir: str,
                 output_base_dir: str):
        """
        Initialize the batch ingestion pipeline.
        
        Args:
            dataset_dir: Directory containing original PDF files
            data_for_finetuning_dir: Directory for new PDFs to be processed
            data_json_dir: Directory for extracted JSON files
            output_base_dir: Base directory for all outputs
        """
        self.dataset_dir = Path(dataset_dir)
        self.data_for_finetuning_dir = Path(data_for_finetuning_dir)
        self.data_json_dir = Path(data_json_dir)
        self.output_base_dir = Path(output_base_dir)
        
        # Create directories
        self.data_for_finetuning_dir.mkdir(exist_ok=True)
        self.data_json_dir.mkdir(exist_ok=True)
        self.output_base_dir.mkdir(exist_ok=True)
        
        # Subdirectories
        self.cleaned_data_dir = self.output_base_dir / "cleaned_data"
        self.chunked_data_dir = self.output_base_dir / "chunked_data"
        self.cleaned_data_dir.mkdir(exist_ok=True)
        self.chunked_data_dir.mkdir(exist_ok=True)
        
        # Initialize processors
        self.pdf_extractor = PDFTextExtractor(str(self.dataset_dir), str(self.data_json_dir))
        self.text_processor = EnvironmentTextProcessor(str(self.data_json_dir), str(self.cleaned_data_dir))
        self.text_chunker = TextChunker(str(self.cleaned_data_dir), str(self.chunked_data_dir))
        
        # Track processed files
        self.processed_files = self._load_processed_files()
    
    def _load_processed_files(self) -> Dict[str, Dict]:
        """Load tracking of already processed files."""
        tracking_file = self.output_base_dir / "processed_files_tracking.json"
        
        if tracking_file.exists():
            try:
                with open(tracking_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load processed files tracking: {e}")
        
        return {"files": {}, "last_updated": None}
    
    def _save_processed_files(self):
        """Save tracking of processed files."""
        tracking_file = self.output_base_dir / "processed_files_tracking.json"
        
        self.processed_files["last_updated"] = datetime.now().isoformat()
        
        try:
            with open(tracking_file, 'w', encoding='utf-8') as f:
                json.dump(self.processed_files, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Could not save processed files tracking: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get MD5 hash of file for change detection."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Could not get hash for {file_path}: {e}")
            return ""
    
    def _is_file_processed(self, file_path: Path) -> bool:
        """Check if file has already been processed."""
        file_hash = self._get_file_hash(file_path)
        file_name = file_path.name
        
        if file_name in self.processed_files["files"]:
            stored_hash = self.processed_files["files"][file_name].get("hash", "")
            return stored_hash == file_hash
        
        return False
    
    def _mark_file_processed(self, file_path: Path, processing_stages: List[str]):
        """Mark file as processed with processing stages."""
        file_hash = self._get_file_hash(file_path)
        file_name = file_path.name
        
        self.processed_files["files"][file_name] = {
            "hash": file_hash,
            "processed_at": datetime.now().isoformat(),
            "stages_completed": processing_stages,
            "file_size": file_path.stat().st_size
        }
    
    def detect_new_files(self) -> List[Path]:
        """Detect new files in data_for_finetuning directory that don't exist in dataset."""
        # Get all files from dataset (original files)
        dataset_files = set()
        if self.dataset_dir.exists():
            for file_path in self.dataset_dir.iterdir():
                if file_path.is_file():
                    dataset_files.add(file_path.name)
        
        # Get all files from data_for_finetuning
        finetuning_files = list(self.data_for_finetuning_dir.glob("*"))
        finetuning_files = [f for f in finetuning_files if f.is_file()]
        
        # Find truly new files (not in dataset)
        new_files = []
        for file_path in finetuning_files:
            if file_path.name not in dataset_files:
                new_files.append(file_path)
                logger.info(f"New file detected: {file_path.name}")
        
        return new_files
    
    def extract_text_from_file(self, file_path: Path) -> Dict:
        """Extract text from various file formats."""
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.pdf':
            return self.pdf_extractor.extract_text_from_pdf(file_path)
        elif file_extension in ['.txt', '.md']:
            return self._extract_text_from_text_file(file_path)
        else:
            return {
                "success": False,
                "error": f"Unsupported file format: {file_extension}",
                "method": "unsupported"
            }
    
    def _extract_text_from_text_file(self, file_path: Path) -> Dict:
        """Extract text from plain text files (TXT, MD)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            return {
                "success": True,
                "full_text": text,
                "total_text_length": len(text),
                "total_pages": 1,  # Text files are treated as single page
                "method": "text_file",
                "pages": [{"page_number": 1, "text": text, "char_count": len(text)}]
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read text file: {str(e)}",
                "method": "text_file"
            }
    
    def process_new_file(self, file_path: Path) -> Dict:
        """
        Process a single new file through the entire pipeline.
        
        Args:
            file_path: Path to file (PDF, TXT, MD, etc.)
            
        Returns:
            Processing results dictionary
        """
        logger.info(f"Processing new file: {file_path.name}")
        
        processing_stages = []
        results = {
            "file": file_path.name,
            "success": False,
            "stages": {},
            "error": None
        }
        
        try:
            # Stage 1: Extract text from file
            logger.info(f"Stage 1: Extracting text from {file_path.name}")
            extraction_result = self.extract_text_from_file(file_path)
            
            if not extraction_result.get("success", False):
                results["error"] = f"File extraction failed: {extraction_result.get('error', 'Unknown error')}"
                return results
            
            # Save extracted text
            json_name = file_path.stem + '.json'  # Remove extension and add .json
            json_path = self.data_json_dir / json_name
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(extraction_result, f, indent=2, ensure_ascii=False)
            
            processing_stages.append("extraction")
            results["stages"]["extraction"] = {
                "success": True,
                "pages": extraction_result.get("total_pages", 0),
                "text_length": extraction_result.get("total_text_length", 0)
            }
            
            # Stage 2: Clean and filter text
            logger.info(f"Stage 2: Cleaning text from {file_path.name}")
            cleaned_text, filtering_stats = self.text_processor.clean_text(
                extraction_result.get("full_text", "")
            )
            
            if not cleaned_text.strip():
                results["error"] = "No relevant content found after filtering"
                return results
            
            # Create cleaned data structure
            cleaned_data = {
                "source_file": file_path.name,
                "processing_timestamp": datetime.now().isoformat(),
                "original_metadata": {
                    "total_pages": extraction_result.get("total_pages", 0),
                    "original_text_length": extraction_result.get("total_text_length", 0),
                    "extraction_method": extraction_result.get("method", "unknown")
                },
                "filtering_stats": filtering_stats,
                "cleaned_text": cleaned_text,
                "processing_success": True
            }
            
            # Save cleaned text
            cleaned_filename = json_name.replace('.json', '_cleaned.json')
            cleaned_path = self.cleaned_data_dir / cleaned_filename
            
            with open(cleaned_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
            
            processing_stages.append("cleaning")
            results["stages"]["cleaning"] = {
                "success": True,
                "retention_rate": filtering_stats.get("retention_rate", 0),
                "cleaned_length": filtering_stats.get("cleaned_length", 0)
            }
            
            # Stage 3: Chunk text
            logger.info(f"Stage 3: Chunking text from {file_path.name}")
            
            # Create chunks for each strategy and size
            all_chunks = {}
            chunk_stats = {}
            
            for size_name, chunk_size in self.text_chunker.chunk_sizes.items():
                overlap_percentage = self.text_chunker.overlap_percentages[size_name]
                
                size_chunks = {}
                for strategy in ["fixed_size", "content_aware", "recursive_character"]:
                    try:
                        chunks = self.text_chunker.chunk_text(
                            cleaned_text, strategy, chunk_size, overlap_percentage
                        )
                        size_chunks[strategy] = chunks
                        
                        # Calculate statistics
                        chunk_stats[f"{size_name}_{strategy}"] = {
                            "chunk_count": len(chunks),
                            "avg_tokens": sum(c["estimated_tokens"] for c in chunks) / len(chunks) if chunks else 0,
                            "min_tokens": min(c["estimated_tokens"] for c in chunks) if chunks else 0,
                            "max_tokens": max(c["estimated_tokens"] for c in chunks) if chunks else 0,
                            "total_tokens": sum(c["estimated_tokens"] for c in chunks)
                        }
                        
                    except Exception as e:
                        logger.error(f"Error chunking {file_path.name} with {strategy}: {str(e)}")
                        size_chunks[strategy] = []
                
                all_chunks[size_name] = size_chunks
            
            # Create chunked data structure
            chunked_data = {
                "source_file": file_path.name,
                "chunking_timestamp": datetime.now().isoformat(),
                "original_metadata": cleaned_data["original_metadata"],
                "filtering_stats": filtering_stats,
                "chunking_config": {
                    "chunk_sizes": self.text_chunker.chunk_sizes,
                    "overlap_percentages": self.text_chunker.overlap_percentages
                },
                "chunk_stats": chunk_stats,
                "chunks": all_chunks,
                "chunking_success": True
            }
            
            # Save chunked data
            chunked_filename = json_name.replace('.json', '_chunked.json')
            chunked_path = self.chunked_data_dir / chunked_filename
            
            with open(chunked_path, 'w', encoding='utf-8') as f:
                json.dump(chunked_data, f, indent=2, ensure_ascii=False)
            
            processing_stages.append("chunking")
            total_chunks = sum(len(chunks) for size_chunks in all_chunks.values() 
                             for chunks in size_chunks.values())
            
            results["stages"]["chunking"] = {
                "success": True,
                "total_chunks": total_chunks,
                "chunk_stats": chunk_stats,
                "chunked_file": str(chunked_path)
            }
            results["chunked_file_path"] = str(chunked_path)
            
            # Mark file as processed
            self._mark_file_processed(file_path, processing_stages)
            
            results["success"] = True
            logger.info(f"Successfully processed {file_path.name} through all stages")
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    def run_batch_processing(self) -> Dict:
        """
        Run the complete batch processing pipeline.
        
        Returns:
            Summary of batch processing results
        """
        logger.info("Starting batch ingestion pipeline...")
        
        # Detect new files
        new_files = self.detect_new_files()
        
        if not new_files:
            logger.info("No new files detected")
            return {
                "new_files_detected": 0,
                "files_processed": 0,
                "processing_summary": "No new files to process"
            }
        
        logger.info(f"Found {len(new_files)} new files to process")
        
        # Process each new file
        results = []
        successful_processing = 0
        
        for file_path in new_files:
            result = self.process_new_file(file_path)
            results.append(result)
            
            if result["success"]:
                successful_processing += 1
        
        # Save tracking information
        self._save_processed_files()
        
        # Generate summary
        summary = {
            "batch_processing_summary": {
                "new_files_detected": len(new_files),
                "files_processed": successful_processing,
                "processing_failed": len(new_files) - successful_processing,
                "processing_timestamp": datetime.now().isoformat()
            },
            "file_results": results
        }
        
        # Save batch processing summary
        summary_path = self.output_base_dir / "batch_processing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch processing complete: {successful_processing}/{len(new_files)} files processed successfully")
        
        return summary
    
    def get_pipeline_status(self) -> Dict:
        """Get current status of the pipeline."""
        # Count files in each stage
        json_files = len(list(self.data_json_dir.glob("*.json")))
        cleaned_files = len(list(self.cleaned_data_dir.glob("*_cleaned.json")))
        chunked_files = len(list(self.chunked_data_dir.glob("*_chunked.json")))
        new_files = len(self.detect_new_files())
        
        return {
            "pipeline_status": {
                "extracted_files": json_files,
                "cleaned_files": cleaned_files,
                "chunked_files": chunked_files,
                "new_files_pending": new_files,
                "last_updated": self.processed_files.get("last_updated", "Never")
            }
        }


def main():
    """Main function to run the batch ingestion pipeline."""
    # Define paths
    dataset_dir = "/media/data/codes/reshma/lma_maj_pro/dataset"
    data_for_finetuning_dir = "/media/data/codes/reshma/lma_maj_pro/data_for_finetuning"
    data_json_dir = "/media/data/codes/reshma/lma_maj_pro/data_json"
    output_base_dir = "/media/data/codes/reshma/lma_maj_pro/partb"
    
    # Create pipeline instance
    pipeline = BatchIngestionPipeline(
        dataset_dir=dataset_dir,
        data_for_finetuning_dir=data_for_finetuning_dir,
        data_json_dir=data_json_dir,
        output_base_dir=output_base_dir
    )
    
    # Get current status
    status = pipeline.get_pipeline_status()
    logger.info(f"Pipeline Status: {status}")
    
    # Run batch processing
    summary = pipeline.run_batch_processing()
    
    logger.info("Batch ingestion pipeline completed!")
    logger.info(f"Summary: {summary}")


if __name__ == "__main__":
    main()
