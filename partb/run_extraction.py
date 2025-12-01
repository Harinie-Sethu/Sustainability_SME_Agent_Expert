#!/usr/bin/env python3
"""
Simple runner script for PDF text extraction.
This script provides an easy way to run the PDF text extraction process.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from pdf_text_extractor import PDFTextExtractor
import logging

def main():
    """Run the PDF text extraction with custom settings."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Define paths
    dataset_dir = "/media/data/codes/reshma/lma_maj_pro/dataset"
    output_dir = "/media/data/codes/reshma/lma_maj_pro/partb"
    
    # Verify dataset directory exists
    if not Path(dataset_dir).exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return 1
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    try:
        # Create extractor and run
        extractor = PDFTextExtractor(dataset_dir, output_dir)
        results = extractor.process_all_pdfs()
        
        # Print summary
        successful = sum(1 for r in results if r.get("success", False))
        total = len(results)
        
        print(f"\n{'='*50}")
        print(f"EXTRACTION COMPLETE")
        print(f"{'='*50}")
        print(f"Total files processed: {total}")
        print(f"Successful extractions: {successful}")
        print(f"Failed extractions: {total - successful}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*50}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
