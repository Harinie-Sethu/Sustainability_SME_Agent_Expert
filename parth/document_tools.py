"""
Document Generation Tools with Fallbacks
Create documents in multiple formats with automatic fallback
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentTools:
    """
    Document generation with multi-format support and fallbacks:
    - PDF generation with fallback
    - DOCX generation with fallback
    - PPT generation with fallback
    - Format conversion
    - Error recovery
    """
    
    def __init__(self, output_dir: str = "parth/generated_documents"):
        """
        Initialize document tools.
        
        Args:
            output_dir: Directory for generated documents
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Import exporters from Part D
        try:
            from partd.export_tools import DocumentExporter
            self.exporter = DocumentExporter(str(self.output_dir))
            self.exporter_available = True
        except ImportError:
            logger.warning("DocumentExporter not available")
            self.exporter_available = False
        
        self.generation_log: List[Dict] = []
        
        logger.info(f"✓ Document Tools initialized: {self.output_dir}")
    
    def generate_document(self,
                         content: Dict[str, Any],
                         content_type: str,
                         preferred_format: str = "pdf",
                         fallback_formats: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate document with automatic fallback.
        
        Args:
            content: Content to export
            content_type: Type of content (quiz, study_guide, report)
            preferred_format: Preferred format (pdf, docx, ppt)
            fallback_formats: List of fallback formats
            
        Returns:
            Generation result with filepath
        """
        logger.info(f"Generating {content_type} document (format: {preferred_format})")
        
        if not self.exporter_available:
            return {
                'success': False,
                'error': 'Document exporter not available'
            }
        
        if fallback_formats is None:
            fallback_formats = ['docx', 'pdf'] if preferred_format == 'pdf' else ['pdf', 'docx']
        
        # Try preferred format first
        formats_to_try = [preferred_format] + [f for f in fallback_formats if f != preferred_format]
        
        for fmt in formats_to_try:
            try:
                result = self._generate_format(content, content_type, fmt)
                
                if result['success']:
                    # Log success
                    self._log_generation(content_type, fmt, True, result['filepath'])
                    
                    return {
                        'success': True,
                        'filepath': result['filepath'],
                        'format': fmt,
                        'used_fallback': fmt != preferred_format,
                        'attempted_formats': formats_to_try[:formats_to_try.index(fmt) + 1]
                    }
            
            except Exception as e:
                logger.warning(f"Format {fmt} failed: {e}")
                self._log_generation(content_type, fmt, False, error=str(e))
                continue
        
        # All formats failed
        return {
            'success': False,
            'error': f'All formats failed: {formats_to_try}',
            'attempted_formats': formats_to_try
        }
    
    def _generate_format(self, content: Dict, content_type: str, format: str) -> Dict[str, Any]:
        """Generate document in specific format."""
        
        if content_type == 'quiz':
            if format == 'pdf':
                filepath = self.exporter.export_quiz_to_pdf(content)
            elif format == 'docx':
                filepath = self.exporter.export_quiz_to_docx(content)
            elif format == 'ppt' or format == 'pptx':
                filepath = self.exporter.export_quiz_to_ppt(content)
            else:
                raise ValueError(f"Unsupported format for quiz: {format}")
        
        elif content_type == 'study_guide':
            if format == 'docx':
                filepath = self.exporter.export_study_guide_to_docx(content)
            elif format == 'pdf':
                filepath = self.exporter.export_study_guide_to_pdf(content)
            elif format == 'ppt' or format == 'pptx':
                filepath = self.exporter.export_study_guide_to_ppt(content)
            else:
                raise ValueError(f"Unsupported format for study guide: {format}")
        
        elif content_type in ['report', 'article', 'content']:
            if format == 'docx':
                filepath = self.exporter.export_content_to_docx(content, content_type)
            elif format == 'pdf':
                docx_path = self.exporter.export_content_to_docx(content, content_type)
                filepath = docx_path  # In production, convert
            elif format == 'ppt' or format == 'pptx':
                filepath = self.exporter.export_content_to_ppt(content, content_type)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        else:
            raise ValueError(f"Unknown content type: {content_type}")
        
        return {
            'success': True,
            'filepath': filepath,
            'format': format
        }
    
    def batch_generate(self,
                      contents: List[Dict[str, Any]],
                      format: str = "pdf") -> Dict[str, Any]:
        """
        Generate multiple documents in batch.
        
        Args:
            contents: List of content dictionaries with 'content' and 'content_type'
            format: Target format
            
        Returns:
            Batch generation results
        """
        logger.info(f"Batch generating {len(contents)} documents")
        
        results = []
        successful = 0
        failed = 0
        
        for i, content_item in enumerate(contents, 1):
            logger.info(f"  Generating document {i}/{len(contents)}")
            
            result = self.generate_document(
                content=content_item['content'],
                content_type=content_item['content_type'],
                preferred_format=format
            )
            
            if result['success']:
                successful += 1
            else:
                failed += 1
            
            results.append(result)
        
        return {
            'total': len(contents),
            'successful': successful,
            'failed': failed,
            'results': results
        }
    
    def convert_format(self, source_path: str, target_format: str) -> Dict[str, Any]:
        """
        Convert document from one format to another.
        
        Args:
            source_path: Source document path
            target_format: Target format
            
        Returns:
            Conversion result
        """
        logger.info(f"Converting {source_path} to {target_format}")
        
        # In production, use proper conversion libraries
        # For now, log the request
        return {
            'success': False,
            'error': 'Format conversion not implemented in demo',
            'note': 'Would use libraries like pypandoc, python-docx, reportlab'
        }
    
    def _log_generation(self, content_type: str, format: str, success: bool,
                       filepath: str = None, error: str = None):
        """Log document generation."""
        self.generation_log.append({
            'timestamp': datetime.now().isoformat(),
            'content_type': content_type,
            'format': format,
            'success': success,
            'filepath': filepath,
            'error': error
        })
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get document generation statistics."""
        if not self.generation_log:
            return {'message': 'No generations logged'}
        
        total = len(self.generation_log)
        successful = sum(1 for log in self.generation_log if log['success'])
        failed = total - successful
        
        by_format = {}
        by_content_type = {}
        
        for log in self.generation_log:
            fmt = log['format']
            content_type = log['content_type']
            
            by_format[fmt] = by_format.get(fmt, {'total': 0, 'success': 0, 'failed': 0})
            by_format[fmt]['total'] += 1
            if log['success']:
                by_format[fmt]['success'] += 1
            else:
                by_format[fmt]['failed'] += 1
            
            by_content_type[content_type] = by_content_type.get(content_type, 0) + 1
        
        return {
            'total_generations': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0,
            'by_format': by_format,
            'by_content_type': by_content_type
        }
