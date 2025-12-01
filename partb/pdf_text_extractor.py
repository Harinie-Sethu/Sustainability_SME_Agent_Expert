#!/usr/bin/env python3
"""
PDF Text Extraction Tool
Extracts text from PDF files and saves them in JSON format for easy text processing.
Each output file maintains the same name as the original PDF file.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import PyPDF2
import fitz  # PyMuPDF
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PDFTextExtractor:
    """Extract text from PDF files using multiple methods for better accuracy."""
    
    def __init__(self, dataset_dir: str, output_dir: str):
        """
        Initialize the PDF text extractor.
        
        Args:
            dataset_dir: Path to directory containing PDF files
            output_dir: Path to directory where extracted text will be saved
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def extract_text_pymupdf(self, pdf_path: Path) -> Dict:
        """
        Extract text using PyMuPDF (fitz) - generally more accurate.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            doc = fitz.open(pdf_path)
            text_content = []
            page_metadata = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                text_content.append(text)
                
                # Get page metadata
                page_info = {
                    "page_number": page_num + 1,
                    "text_length": len(text),
                    "has_text": len(text.strip()) > 0
                }
                page_metadata.append(page_info)
            
            doc.close()
            
            return {
                "method": "pymupdf",
                "success": True,
                "total_pages": len(doc),
                "pages": page_metadata,
                "full_text": "\n\n".join(text_content),
                "total_text_length": sum(len(text) for text in text_content)
            }
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for {pdf_path}: {str(e)}")
            return {
                "method": "pymupdf",
                "success": False,
                "error": str(e)
            }
    
    def extract_text_pypdf2(self, pdf_path: Path) -> Dict:
        """
        Extract text using PyPDF2 as fallback method.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = []
                page_metadata = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    text_content.append(text)
                    
                    page_info = {
                        "page_number": page_num + 1,
                        "text_length": len(text),
                        "has_text": len(text.strip()) > 0
                    }
                    page_metadata.append(page_info)
                
                return {
                    "method": "pypdf2",
                    "success": True,
                    "total_pages": len(pdf_reader.pages),
                    "pages": page_metadata,
                    "full_text": "\n\n".join(text_content),
                    "total_text_length": sum(len(text) for text in text_content)
                }
                
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed for {pdf_path}: {str(e)}")
            return {
                "method": "pypdf2",
                "success": False,
                "error": str(e)
            }
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict:
        """
        Extract text from PDF using the best available method.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        logger.info(f"Extracting text from: {pdf_path.name}")
        
        # Try PyMuPDF first (generally more accurate)
        result = self.extract_text_pymupdf(pdf_path)
        
        # If PyMuPDF fails, try PyPDF2
        if not result["success"]:
            logger.warning(f"PyMuPDF failed for {pdf_path.name}, trying PyPDF2...")
            result = self.extract_text_pypdf2(pdf_path)
        
        # Add file metadata
        result.update({
            "source_file": pdf_path.name,
            "extraction_timestamp": datetime.now().isoformat(),
            "file_size_bytes": pdf_path.stat().st_size
        })
        
        return result
    
    def save_extracted_text(self, pdf_name: str, extraction_result: Dict) -> Path:
        """
        Save extracted text to JSON file with same name as PDF.
        
        Args:
            pdf_name: Name of the original PDF file
            extraction_result: Dictionary containing extracted text and metadata
            
        Returns:
            Path to the saved JSON file
        """
        # Remove .pdf extension and add .json
        json_name = pdf_name.replace('.pdf', '.json')
        output_path = self.output_dir / json_name
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(extraction_result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved extracted text to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save {output_path}: {str(e)}")
            raise
    
    def process_all_pdfs(self) -> List[Dict]:
        """
        Process all PDF files in the dataset directory.
        
        Returns:
            List of processing results for each PDF
        """
        pdf_files = list(self.dataset_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.dataset_dir}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        results = []
        successful_extractions = 0
        
        for pdf_path in pdf_files:
            try:
                # Extract text
                extraction_result = self.extract_text_from_pdf(pdf_path)
                
                # Save to JSON file
                output_path = self.save_extracted_text(pdf_path.name, extraction_result)
                
                result = {
                    "pdf_file": pdf_path.name,
                    "output_file": output_path.name,
                    "success": extraction_result["success"],
                    "pages_extracted": extraction_result.get("total_pages", 0),
                    "text_length": extraction_result.get("total_text_length", 0)
                }
                
                if extraction_result["success"]:
                    successful_extractions += 1
                
                results.append(result)
                logger.info(f"Processed {pdf_path.name}: {result['pages_extracted']} pages, "
                           f"{result['text_length']} characters")
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {str(e)}")
                results.append({
                    "pdf_file": pdf_path.name,
                    "success": False,
                    "error": str(e)
                })
        
        logger.info(f"Processing complete: {successful_extractions}/{len(pdf_files)} files processed successfully")
        return results
    
    def generate_summary_report(self, results: List[Dict]) -> Path:
        """
        Generate a summary report of the extraction process.
        
        Args:
            results: List of processing results
            
        Returns:
            Path to the summary report file
        """
        summary = {
            "extraction_summary": {
                "total_files": len(results),
                "successful_extractions": sum(1 for r in results if r.get("success", False)),
                "failed_extractions": sum(1 for r in results if not r.get("success", False)),
                "total_pages": sum(r.get("pages_extracted", 0) for r in results),
                "total_characters": sum(r.get("text_length", 0) for r in results),
                "extraction_timestamp": datetime.now().isoformat()
            },
            "file_results": results
        }
        
        summary_path = self.output_dir / "extraction_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary report saved to: {summary_path}")
        return summary_path


def main():
    """Main function to run the PDF text extraction process."""
    # Define paths
    dataset_dir = "/media/data/codes/reshma/lma_maj_pro/dataset"
    output_dir = "/media/data/codes/reshma/lma_maj_pro/partb"
    
    # Create extractor instance
    extractor = PDFTextExtractor(dataset_dir, output_dir)
    
    # Process all PDFs
    logger.info("Starting PDF text extraction process...")
    results = extractor.process_all_pdfs()
    
    # Generate summary report
    summary_path = extractor.generate_summary_report(results)
    
    logger.info("PDF text extraction process completed!")
    logger.info(f"Check the output directory: {output_dir}")
    logger.info(f"Summary report: {summary_path}")


if __name__ == "__main__":
    main()
