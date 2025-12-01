"""
Email Tools with Retry Logic
Send emails with automatic retry and fallback
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailTools:
    """
    Email automation with:
    - Retry logic
    - Batch sending
    - Template support
    - Delivery tracking
    - Error recovery
    """
    
    def __init__(self, smtp_config: Optional[Dict] = None):
        """
        Initialize email tools.
        
        Args:
            smtp_config: SMTP configuration
        """
        self.smtp_config = smtp_config or {}
        
        # Import email tool from Part D
        try:
            from partd.email_tool import EmailTool
            
            self.email_client = EmailTool(
                smtp_server=self.smtp_config.get('smtp_server', 'smtp.gmail.com'),
                smtp_port=self.smtp_config.get('smtp_port', 587),
                sender_email=self.smtp_config.get('sender_email'),
                sender_password=self.smtp_config.get('sender_password')
            )
            self.email_available = self.email_client.enabled
        except ImportError:
            logger.warning("Email tool not available")
            self.email_available = False
            self.email_client = None
        
        self.send_log: List[Dict] = []
        
        logger.info(f" Email Tools initialized (available: {self.email_available})")
    
    def send_with_retry(self,
                       recipient: str,
                       subject: str,
                       body: str,
                       attachments: Optional[List[str]] = None,
                       max_retries: int = 3) -> Dict[str, Any]:
        """
        Send email with retry logic.
        
        Args:
            recipient: Recipient email
            subject: Email subject
            body: Email body
            attachments: List of attachment paths
            max_retries: Maximum retry attempts
            
        Returns:
            Send result
        """
        logger.info(f"Sending email to {recipient} (max_retries={max_retries})")
        
        if not self.email_available:
            return {
                'success': False,
                'error': 'Email service not configured',
                'note': 'Set SENDER_EMAIL and SENDER_PASSWORD environment variables'
            }
        
        attempt = 0
        last_error = None
        
        while attempt < max_retries:
            try:
                success = self.email_client.send_document(
                    recipient=recipient,
                    subject=subject,
                    body=body,
                    attachments=attachments or []
                )
                
                if success:
                    self._log_send(recipient, subject, True, attempt + 1)
                    
                    return {
                        'success': True,
                        'recipient': recipient,
                        'attempts': attempt + 1
                    }
                else:
                    raise Exception("Email send returned False")
            
            except Exception as e:
                attempt += 1
                last_error = e
                
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} attempts failed")
        
        self._log_send(recipient, subject, False, max_retries, str(last_error))
        
        return {
            'success': False,
            'error': str(last_error),
            'attempts': max_retries
        }
    
    def send_batch(self,
                  recipients: List[str],
                  subject: str,
                  body: str,
                  attachments: Optional[List[str]] = None,
                  max_retries: int = 3) -> Dict[str, Any]:
        """
        Send email to multiple recipients.
        
        Args:
            recipients: List of recipient emails
            subject: Email subject
            body: Email body
            attachments: List of attachment paths
            max_retries: Maximum retry attempts per recipient
            
        Returns:
            Batch send results
        """
        logger.info(f"Batch sending to {len(recipients)} recipients")
        
        results = []
        successful = 0
        failed = 0
        
        for i, recipient in enumerate(recipients, 1):
            logger.info(f"  Sending to recipient {i}/{len(recipients)}: {recipient}")
            
            result = self.send_with_retry(
                recipient=recipient,
                subject=subject,
                body=body,
                attachments=attachments,
                max_retries=max_retries
            )
            
            if result['success']:
                successful += 1
            else:
                failed += 1
            
            results.append(result)
            
            # Brief delay between sends to avoid rate limiting
            if i < len(recipients):
                time.sleep(1)
        
        return {
            'total': len(recipients),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(recipients) if recipients else 0,
            'results': results
        }
    
    def send_with_template(self,
                          recipient: str,
                          template_name: str,
                          template_vars: Dict[str, Any],
                          attachments: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Send email using template.
        
        Args:
            recipient: Recipient email
            template_name: Template name
            template_vars: Variables for template
            attachments: Attachments
            
        Returns:
            Send result
        """
        # Load template
        subject, body = self._load_template(template_name, template_vars)
        
        return self.send_with_retry(
            recipient=recipient,
            subject=subject,
            body=body,
            attachments=attachments
        )
    
    def _load_template(self, template_name: str, variables: Dict[str, Any]) -> tuple:
        """Load and format email template."""
        
        # Define templates
        templates = {
            'quiz_delivery': {
                'subject': 'Quiz: {topic}',
                'body': """Hello,

Please find attached the quiz on {topic}.

Difficulty: {difficulty}
Number of Questions: {num_questions}

Best regards,
Environmental Sustainability System"""
            },
            'study_guide': {
                'subject': 'Study Guide: {topic}',
                'body': """Hello,

Attached is the comprehensive study guide on {topic}.

This guide covers key concepts, facts, and actionable insights.

Best regards,
Environmental Sustainability System"""
            },
            'report_delivery': {
                'subject': 'Report: {title}',
                'body': """Hello,

Please find attached the report: {title}

Generated on: {date}

Best regards,
Environmental Sustainability System"""
            }
        }
        
        template = templates.get(template_name)
        if not template:
            return "Document Delivery", "Please find the attached document."
        
        subject = template['subject'].format(**variables)
        body = template['body'].format(**variables)
        
        return subject, body
    
    def _log_send(self, recipient: str, subject: str, success: bool,
                 attempts: int, error: str = None):
        """Log email send."""
        self.send_log.append({
            'timestamp': datetime.now().isoformat(),
            'recipient': recipient,
            'subject': subject,
            'success': success,
            'attempts': attempts,
            'error': error
        })
    
    def get_send_statistics(self) -> Dict[str, Any]:
        """Get email send statistics."""
        if not self.send_log:
            return {'message': 'No sends logged'}
        
        total = len(self.send_log)
        successful = sum(1 for log in self.send_log if log['success'])
        failed = total - successful
        
        avg_attempts = sum(log['attempts'] for log in self.send_log) / total
        
        return {
            'total_sends': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0,
            'avg_attempts': avg_attempts,
            'max_attempts': max(log['attempts'] for log in self.send_log)
        }
