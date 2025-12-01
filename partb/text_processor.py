#!/usr/bin/env python3
"""
Text Processing Pipeline for Environment/Sustainability Documents
Performs semantic filtering, cleaning, and preprocessing of extracted PDF text.
"""

import os
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import unicodedata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnvironmentTextProcessor:
    """Process and clean text for environment/sustainability content."""
    
    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize the text processor.
        
        Args:
            data_dir: Directory containing JSON files with extracted text
            output_dir: Directory to save processed text
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Environment/Sustainability keywords for semantic filtering
        self.environment_keywords = {
            # Core environmental terms
            'environment', 'environmental', 'ecology', 'ecological', 'ecosystem', 'ecosystems',
            'sustainability', 'sustainable', 'conservation', 'preservation', 'biodiversity',
            'climate', 'climate change', 'global warming', 'greenhouse', 'carbon', 'emissions',
            'pollution', 'contamination', 'waste', 'recycling', 'renewable', 'energy',
            'natural resources', 'deforestation', 'habitat', 'wildlife', 'species',
            
            # Environmental education and awareness
            'environmental education', 'environmental awareness', 'environmental literacy',
            'environmental studies', 'environmental science', 'environmental policy',
            'environmental impact', 'environmental assessment', 'environmental management',
            
            # Sustainability concepts
            'sustainable development', 'sustainable practices', 'sustainable living',
            'green technology', 'green energy', 'green building', 'green economy',
            'circular economy', 'zero waste', 'carbon footprint', 'carbon neutral',
            
            # Environmental issues
            'air quality', 'water quality', 'soil degradation', 'ocean acidification',
            'ozone depletion', 'acid rain', 'urbanization', 'industrial pollution',
            'agricultural impact', 'fishing impact', 'mining impact',
            
            # Environmental solutions
            'renewable energy', 'solar power', 'wind energy', 'hydroelectric',
            'clean technology', 'environmental restoration', 'reforestation',
            'environmental monitoring', 'environmental protection', 'environmental law',
            
            # Related academic terms
            'environmental research', 'environmental analysis', 'environmental planning',
            'environmental ethics', 'environmental justice', 'environmental health',
            'environmental economics', 'environmental sociology', 'environmental psychology'
        }
        
        # Compile regex patterns for efficient matching
        self.keyword_patterns = self._compile_keyword_patterns()
        
    def _compile_keyword_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for keyword matching."""
        patterns = []
        
        # Create patterns for different keyword categories
        for keyword in self.environment_keywords:
            # Case-insensitive pattern with word boundaries
            pattern = re.compile(r'\b' + re.escape(keyword.lower()) + r'\b', re.IGNORECASE)
            patterns.append(pattern)
            
        return patterns
    
    def clean_text(self, text: str) -> Tuple[str, Dict]:
        """
        Clean and filter text for environment/sustainability relevance.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Tuple of (cleaned_text, filtering_stats)
        """
        if not text or not text.strip():
            return "", {"original_length": 0, "cleaned_length": 0, "sections_removed": 0}
        
        original_length = len(text)
        filtering_stats = {
            "original_length": original_length,
            "sections_removed": 0,
            "paragraphs_removed": 0,
            "sentences_removed": 0
        }
        
        try:
            # Step 1: Basic text normalization
            cleaned_text = self._normalize_text(text)
            
            # Step 2: Split into sections/paragraphs for semantic filtering
            sections = self._split_into_sections(cleaned_text)
            
            # Step 3: Filter sections based on environment/sustainability relevance
            relevant_sections = []
            for section in sections:
                if self._is_environmentally_relevant(section):
                    relevant_sections.append(section)
                else:
                    filtering_stats["sections_removed"] += 1
            
            # Step 4: Further clean remaining text
            if relevant_sections:
                cleaned_text = "\n\n".join(relevant_sections)
                cleaned_text = self._final_cleanup(cleaned_text)
            else:
                cleaned_text = ""
            
            filtering_stats["cleaned_length"] = len(cleaned_text)
            filtering_stats["retention_rate"] = (
                filtering_stats["cleaned_length"] / original_length * 100 
                if original_length > 0 else 0
            )
            
            return cleaned_text, filtering_stats
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text, {"error": str(e), "original_length": original_length}
    
    def _normalize_text(self, text: str) -> str:
        """Basic text normalization."""
        # Convert to lowercase
        text = text.lower()
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        return text.strip()
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into logical sections (paragraphs, sections, etc.)."""
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Also split by common section markers
        sections = []
        for para in paragraphs:
            if para.strip():
                # Split by common section headers
                sub_sections = re.split(r'\n\s*(?:chapter|section|part|unit)\s*\d+', para, flags=re.IGNORECASE)
                sections.extend([s.strip() for s in sub_sections if s.strip()])
        
        return [s for s in sections if len(s.strip()) > 50]  # Filter very short sections
    
    def _is_environmentally_relevant(self, text: str) -> bool:
        """Check if text section is environmentally relevant."""
        if len(text.strip()) < 100:  # Skip very short sections
            return False
        
        # Count keyword matches
        keyword_count = 0
        for pattern in self.keyword_patterns:
            matches = pattern.findall(text)
            keyword_count += len(matches)
        
        # Consider relevant if it contains at least 2 environment-related keywords
        # or if it's a substantial section with at least 1 keyword
        return keyword_count >= 2 or (len(text) > 500 and keyword_count >= 1)
    
    def _final_cleanup(self, text: str) -> str:
        """Final cleanup of processed text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove standalone numbers and single characters
        text = re.sub(r'\b\w\b', '', text)
        
        # Clean up punctuation
        text = re.sub(r'\s+([,\.!?;:])', r'\1', text)
        
        return text.strip()
    
    def process_json_file(self, json_path: Path) -> Dict:
        """
        Process a single JSON file.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Processing results dictionary
        """
        logger.info(f"Processing: {json_path.name}")
        
        try:
            # Load JSON data
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data.get("success", False):
                return {
                    "file": json_path.name,
                    "success": False,
                    "error": "Original extraction failed"
                }
            
            # Extract and clean text
            original_text = data.get("full_text", "")
            cleaned_text, filtering_stats = self.clean_text(original_text)
            
            # Create processed data structure
            processed_data = {
                "source_file": data.get("source_file", json_path.name),
                "processing_timestamp": datetime.now().isoformat(),
                "original_metadata": {
                    "total_pages": data.get("total_pages", 0),
                    "original_text_length": data.get("total_text_length", 0),
                    "extraction_method": data.get("method", "unknown")
                },
                "filtering_stats": filtering_stats,
                "cleaned_text": cleaned_text,
                "processing_success": True
            }
            
            # Save processed file
            output_filename = json_path.name.replace('.json', '_cleaned.json')
            output_path = self.output_dir / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Processed {json_path.name}: {filtering_stats.get('retention_rate', 0):.1f}% retained")
            
            return {
                "file": json_path.name,
                "success": True,
                "output_file": output_filename,
                "retention_rate": filtering_stats.get('retention_rate', 0),
                "original_length": filtering_stats.get('original_length', 0),
                "cleaned_length": filtering_stats.get('cleaned_length', 0)
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
        Process all JSON files in the data directory.
        
        Returns:
            Summary of processing results
        """
        json_files = list(self.data_dir.glob("*.json"))
        
        if not json_files:
            logger.warning(f"No JSON files found in {self.data_dir}")
            return {"error": "No files to process"}
        
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        results = []
        successful_processing = 0
        total_original_length = 0
        total_cleaned_length = 0
        
        for json_file in json_files:
            result = self.process_json_file(json_file)
            results.append(result)
            
            if result.get("success", False):
                successful_processing += 1
                total_original_length += result.get("original_length", 0)
                total_cleaned_length += result.get("cleaned_length", 0)
        
        # Generate summary
        summary = {
            "processing_summary": {
                "total_files": len(json_files),
                "successful_processing": successful_processing,
                "failed_processing": len(json_files) - successful_processing,
                "total_original_length": total_original_length,
                "total_cleaned_length": total_cleaned_length,
                "overall_retention_rate": (
                    total_cleaned_length / total_original_length * 100 
                    if total_original_length > 0 else 0
                ),
                "processing_timestamp": datetime.now().isoformat()
            },
            "file_results": results
        }
        
        # Save summary
        summary_path = self.output_dir / "processing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processing complete: {successful_processing}/{len(json_files)} files processed successfully")
        logger.info(f"Overall retention rate: {summary['processing_summary']['overall_retention_rate']:.1f}%")
        
        return summary


def main():
    """Main function to run text processing."""
    # Define paths
    data_dir = "/media/data/codes/reshma/lma_maj_pro/data_json"
    output_dir = "/media/data/codes/reshma/lma_maj_pro/partb/cleaned_data"
    
    # Create processor instance
    processor = EnvironmentTextProcessor(data_dir, output_dir)
    
    # Process all files
    logger.info("Starting text processing pipeline...")
    summary = processor.process_all_files()
    
    logger.info("Text processing pipeline completed!")
    logger.info(f"Check the output directory: {output_dir}")


if __name__ == "__main__":
    main()
