#!/usr/bin/env python3
"""
Text Chunking Pipeline for Environment/Sustainability Documents
Implements multiple chunking strategies with multi-granularity segmentation.
"""

import os
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import math

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


class TextChunker:
    """Implement multiple chunking strategies for text processing."""
    
    def __init__(self, cleaned_data_dir: str, output_dir: str):
        """
        Initialize the text chunker.
        
        Args:
            cleaned_data_dir: Directory containing cleaned JSON files
            output_dir: Directory to save chunked files
        """
        self.cleaned_data_dir = Path(cleaned_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Token estimation: ~4 characters per token (rough estimate)
        self.chars_per_token = 4
        
        # Chunking configurations
        self.chunk_sizes = {
            "large": 2048,    # Large chunks for context
            "medium": 512,     # Medium chunks for balance
            "small": 128       # Small chunks for precision
        }
        
        # Overlap configurations (percentage)
        self.overlap_percentages = {
            "large": 0.10,     # 10% overlap for large chunks
            "medium": 0.15,    # 15% overlap for medium chunks
            "small": 0.20      # 20% overlap for small chunks
        }
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from character count."""
        return len(text) // self.chars_per_token
    
    def fixed_size_chunking(self, text: str, chunk_size_tokens: int, overlap_percentage: float) -> List[Dict]:
        """
        Fixed-size chunking strategy.
        
        Args:
            text: Text to chunk
            chunk_size_tokens: Target chunk size in tokens
            overlap_percentage: Overlap percentage between chunks
            
        Returns:
            List of chunk dictionaries
        """
        if not text.strip():
            return []
        
        chunk_size_chars = chunk_size_tokens * self.chars_per_token
        overlap_chars = int(chunk_size_chars * overlap_percentage)
        step_size = chunk_size_chars - overlap_chars
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + chunk_size_chars, len(text))
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "strategy": "fixed_size",
                    "chunk_size_tokens": chunk_size_tokens,
                    "overlap_percentage": overlap_percentage,
                    "start_char": start,
                    "end_char": end,
                    "text_length": len(chunk_text),
                    "estimated_tokens": self.estimate_tokens(chunk_text),
                    "text": chunk_text
                })
                chunk_id += 1
            
            start += step_size
        
        return chunks
    
    def content_aware_chunking(self, text: str, chunk_size_tokens: int, overlap_percentage: float) -> List[Dict]:
        """
        Content-aware chunking strategy (paragraph/section-based).
        
        Args:
            text: Text to chunk
            chunk_size_tokens: Target chunk size in tokens
            overlap_percentage: Overlap percentage between chunks
            
        Returns:
            List of chunk dictionaries
        """
        if not text.strip():
            return []
        
        # Split text into paragraphs/sections
        paragraphs = self._split_into_paragraphs(text)
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        start_char = 0
        
        for para in paragraphs:
            para_text = para.strip()
            if not para_text:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            test_chunk = current_chunk + "\n\n" + para_text if current_chunk else para_text
            test_tokens = self.estimate_tokens(test_chunk)
            
            if test_tokens > chunk_size_tokens and current_chunk:
                # Save current chunk
                chunks.append({
                    "chunk_id": chunk_id,
                    "strategy": "content_aware",
                    "chunk_size_tokens": chunk_size_tokens,
                    "overlap_percentage": overlap_percentage,
                    "start_char": start_char,
                    "end_char": start_char + len(current_chunk),
                    "text_length": len(current_chunk),
                    "estimated_tokens": self.estimate_tokens(current_chunk),
                    "text": current_chunk.strip()
                })
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, overlap_percentage)
                current_chunk = overlap_text + "\n\n" + para_text if overlap_text else para_text
                start_char = start_char + len(current_chunk) - len(para_text) - len(overlap_text) - 2
                chunk_id += 1
            else:
                # Add paragraph to current chunk
                current_chunk = test_chunk
                if not start_char:
                    start_char = text.find(para_text)
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "chunk_id": chunk_id,
                "strategy": "content_aware",
                "chunk_size_tokens": chunk_size_tokens,
                "overlap_percentage": overlap_percentage,
                "start_char": start_char,
                "end_char": start_char + len(current_chunk),
                "text_length": len(current_chunk),
                "estimated_tokens": self.estimate_tokens(current_chunk),
                "text": current_chunk.strip()
            })
        
        return chunks
    
    def recursive_character_splitting(self, text: str, chunk_size_tokens: int, overlap_percentage: float) -> List[Dict]:
        """
        Recursive character splitting strategy (iterative implementation to avoid recursion depth issues).
        
        Args:
            text: Text to chunk
            chunk_size_tokens: Target chunk size in tokens
            overlap_percentage: Overlap percentage between chunks
            
        Returns:
            List of chunk dictionaries
        """
        if not text.strip():
            return []
        
        chunk_size_chars = chunk_size_tokens * self.chars_per_token
        chunks = []
        current_pos = 0
        chunk_id = 0
        
        while current_pos < len(text):
            # Get current segment
            remaining_text = text[current_pos:]
            
            if len(remaining_text) <= chunk_size_chars:
                # Last chunk
                chunk_text = remaining_text.strip()
                if chunk_text:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "strategy": "recursive_character",
                        "chunk_size_tokens": chunk_size_tokens,
                        "overlap_percentage": overlap_percentage,
                        "start_char": current_pos,
                        "end_char": current_pos + len(chunk_text),
                        "text_length": len(chunk_text),
                        "estimated_tokens": self.estimate_tokens(chunk_text),
                        "text": chunk_text
                    })
                break
            
            # Find best split point
            split_point = self._find_best_split_point(remaining_text, chunk_size_chars)
            
            if split_point == -1:
                # Fallback to character-based split
                split_point = chunk_size_chars
            
            chunk_text = remaining_text[:split_point].strip()
            
            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "strategy": "recursive_character",
                    "chunk_size_tokens": chunk_size_tokens,
                    "overlap_percentage": overlap_percentage,
                    "start_char": current_pos,
                    "end_char": current_pos + len(chunk_text),
                    "text_length": len(chunk_text),
                    "estimated_tokens": self.estimate_tokens(chunk_text),
                    "text": chunk_text
                })
            
            # Move position with overlap
            overlap_chars = int(chunk_size_chars * overlap_percentage)
            current_pos += max(1, split_point - overlap_chars)
            chunk_id += 1
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split by double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Also split by common section markers
        sections = []
        for para in paragraphs:
            if para.strip():
                # Split by sentence boundaries for very long paragraphs
                if len(para) > 2000:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    sections.extend(sentences)
                else:
                    sections.append(para)
        
        return [s.strip() for s in sections if s.strip()]
    
    def _find_best_split_point(self, text: str, target_length: int) -> int:
        """Find the best split point near the target length."""
        if len(text) <= target_length:
            return len(text)
        
        # Look for sentence boundaries near target length
        search_start = max(0, target_length - 200)
        search_end = min(len(text), target_length + 200)
        
        search_text = text[search_start:search_end]
        
        # Find sentence endings
        sentence_endings = []
        for match in re.finditer(r'[.!?]\s+', search_text):
            sentence_endings.append(search_start + match.end())
        
        if sentence_endings:
            # Find the sentence ending closest to target length
            best_split = min(sentence_endings, key=lambda x: abs(x - target_length))
            return best_split
        
        # Look for paragraph boundaries
        para_boundaries = []
        for match in re.finditer(r'\n\s*\n', search_text):
            para_boundaries.append(search_start + match.end())
        
        if para_boundaries:
            best_split = min(para_boundaries, key=lambda x: abs(x - target_length))
            return best_split
        
        return -1  # No good split point found
    
    def _get_overlap_text(self, text: str, overlap_percentage: float) -> str:
        """Get overlap text from the end of the current chunk."""
        if not text or overlap_percentage <= 0:
            return ""
        
        overlap_length = int(len(text) * overlap_percentage)
        overlap_text = text[-overlap_length:].strip()
        
        # Try to end at a sentence boundary
        last_sentence_end = overlap_text.rfind('.')
        if last_sentence_end > overlap_length * 0.5:  # If sentence boundary is not too far back
            overlap_text = overlap_text[:last_sentence_end + 1]
        
        return overlap_text
    
    def chunk_text(self, text: str, strategy: str, chunk_size_tokens: int, overlap_percentage: float) -> List[Dict]:
        """
        Chunk text using specified strategy.
        
        Args:
            text: Text to chunk
            strategy: Chunking strategy ('fixed_size', 'content_aware', 'recursive_character')
            chunk_size_tokens: Target chunk size in tokens
            overlap_percentage: Overlap percentage between chunks
            
        Returns:
            List of chunk dictionaries
        """
        if strategy == "fixed_size":
            return self.fixed_size_chunking(text, chunk_size_tokens, overlap_percentage)
        elif strategy == "content_aware":
            return self.content_aware_chunking(text, chunk_size_tokens, overlap_percentage)
        elif strategy == "recursive_character":
            return self.recursive_character_splitting(text, chunk_size_tokens, overlap_percentage)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def process_json_file(self, json_path: Path) -> Dict:
        """
        Process a single cleaned JSON file and create chunks.
        
        Args:
            json_path: Path to cleaned JSON file
            
        Returns:
            Processing results dictionary
        """
        logger.info(f"Chunking: {json_path.name}")
        
        try:
            # Load cleaned data
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data.get("processing_success", False):
                return {
                    "file": json_path.name,
                    "success": False,
                    "error": "Original processing failed"
                }
            
            text = data.get("cleaned_text", "")
            if not text.strip():
                return {
                    "file": json_path.name,
                    "success": False,
                    "error": "No text content to chunk"
                }
            
            # Create chunks for each strategy and size
            all_chunks = {}
            chunk_stats = {}
            
            for size_name, chunk_size in self.chunk_sizes.items():
                overlap_percentage = self.overlap_percentages[size_name]
                
                size_chunks = {}
                for strategy in ["fixed_size", "content_aware", "recursive_character"]:
                    try:
                        chunks = self.chunk_text(text, strategy, chunk_size, overlap_percentage)
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
                        logger.error(f"Error chunking {json_path.name} with {strategy}: {str(e)}")
                        size_chunks[strategy] = []
                
                all_chunks[size_name] = size_chunks
            
            # Create chunked data structure
            chunked_data = {
                "source_file": data.get("source_file", json_path.name),
                "chunking_timestamp": datetime.now().isoformat(),
                "original_metadata": data.get("original_metadata", {}),
                "filtering_stats": data.get("filtering_stats", {}),
                "chunking_config": {
                    "chunk_sizes": self.chunk_sizes,
                    "overlap_percentages": self.overlap_percentages
                },
                "chunk_stats": chunk_stats,
                "chunks": all_chunks,
                "chunking_success": True
            }
            
            # Save chunked file
            output_filename = json_path.name.replace('_cleaned.json', '_chunked.json')
            output_path = self.output_dir / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunked_data, f, indent=2, ensure_ascii=False)
            
            total_chunks = sum(len(chunks) for size_chunks in all_chunks.values() 
                             for chunks in size_chunks.values())
            
            logger.info(f"Chunked {json_path.name}: {total_chunks} total chunks created")
            
            return {
                "file": json_path.name,
                "success": True,
                "output_file": output_filename,
                "total_chunks": total_chunks,
                "chunk_stats": chunk_stats
            }
            
        except Exception as e:
            logger.error(f"Error processing {json_path.name}: {str(e)}")
            return {
                "file": json_path.name,
                "success": False,
                "error": str(e)
            }
    
    def process_all_files(self) -> Dict:
        """
        Process all cleaned JSON files and create chunks.
        
        Returns:
            Summary of processing results
        """
        cleaned_files = list(self.cleaned_data_dir.glob("*_cleaned.json"))
        
        if not cleaned_files:
            logger.warning(f"No cleaned JSON files found in {self.cleaned_data_dir}")
            return {"error": "No files to process"}
        
        logger.info(f"Found {len(cleaned_files)} cleaned JSON files to chunk")
        
        results = []
        successful_processing = 0
        total_chunks_created = 0
        
        for json_file in cleaned_files:
            result = self.process_json_file(json_file)
            results.append(result)
            
            if result.get("success", False):
                successful_processing += 1
                total_chunks_created += result.get("total_chunks", 0)
        
        # Generate summary
        summary = {
            "chunking_summary": {
                "total_files": len(cleaned_files),
                "successful_processing": successful_processing,
                "failed_processing": len(cleaned_files) - successful_processing,
                "total_chunks_created": total_chunks_created,
                "avg_chunks_per_file": total_chunks_created / successful_processing if successful_processing > 0 else 0,
                "chunking_timestamp": datetime.now().isoformat()
            },
            "file_results": results
        }
        
        # Save summary
        summary_path = self.output_dir / "chunking_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Chunking complete: {successful_processing}/{len(cleaned_files)} files processed successfully")
        logger.info(f"Total chunks created: {total_chunks_created}")
        
        return summary


def main():
    """Main function to run text chunking."""
    # Define paths
    cleaned_data_dir = "/media/data/codes/reshma/lma_maj_pro/partb/cleaned_data"
    output_dir = "/media/data/codes/reshma/lma_maj_pro/partb/chunked_data"
    
    # Create chunker instance
    chunker = TextChunker(cleaned_data_dir, output_dir)
    
    # Process all files
    logger.info("Starting text chunking pipeline...")
    summary = chunker.process_all_files()
    
    logger.info("Text chunking pipeline completed!")
    logger.info(f"Check the output directory: {output_dir}")


if __name__ == "__main__":
    main()
