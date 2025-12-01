"""
Email Tool for sending documents and reports
Integrated with LangChain multi-step reasoning
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailTool:
    """Send documents via email with LangChain-powered content generation."""
    
    def __init__(self, smtp_server: str = "smtp.gmail.com", 
                 smtp_port: int = 587,
                 sender_email: Optional[str] = None,
                 sender_password: Optional[str] = None,
                 llm_client = None):
        """
        Initialize email tool.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP port
            sender_email: Sender email address
            sender_password: Sender email password/app password
            llm_client: LLM client for generating email content
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email or os.getenv("SENDER_EMAIL")
        self.sender_password = sender_password or os.getenv("SENDER_PASSWORD")
        self.llm_client = llm_client
        
        if not self.sender_email or not self.sender_password:
            logger.warning(" Email credentials not provided. Email functionality disabled.")
            logger.warning("Set SENDER_EMAIL and SENDER_PASSWORD environment variables to enable.")
            self.enabled = False
        else:
            self.enabled = True
            logger.info(f" Email tool initialized: {self.sender_email}")
    
    def send_document(self, recipient: str, subject: str, 
                     body: str, attachments: List[str],
                     cc: Optional[List[str]] = None) -> bool:
        """
        Send email with document attachments.
        
        Args:
            recipient: Recipient email address
            subject: Email subject
            body: Email body text
            attachments: List of file paths to attach
            cc: Optional CC recipients
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            logger.error(" Email tool not enabled. Set credentials first.")
            logger.info(" Tip: Set SENDER_EMAIL and SENDER_PASSWORD environment variables")
            logger.info(f"Would have sent to: {recipient}")
            logger.info(f"Subject: {subject}")
            logger.info(f"Attachments: {attachments}")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient
            msg['Subject'] = subject
            msg['Date'] = datetime.now().strftime("%a, %d %b %Y %H:%M:%S")
            
            if cc:
                msg['Cc'] = ', '.join(cc)
            
            # Add body
            msg.attach(MIMEText(body, 'plain'))
            
            # Add attachments
            for filepath in attachments:
                file_path_obj = Path(filepath)
                if not file_path_obj.exists():
                    logger.warning(f" Attachment not found: {filepath}")
                    continue
                
                # Verify file is not empty
                file_size = file_path_obj.stat().st_size
                if file_size == 0:
                    logger.warning(f" Attachment is empty: {filepath}")
                    continue
                
                try:
                    with open(filepath, 'rb') as f:
                        file_data = f.read()
                        # Verify we actually read data
                        if len(file_data) == 0:
                            logger.warning(f" Attachment read as empty: {filepath}")
                            continue
                        
                        attachment = MIMEApplication(file_data)
                        filename = file_path_obj.name
                        # Set proper content type based on file extension
                        if filename.lower().endswith('.pdf'):
                            attachment.add_header('Content-Type', 'application/pdf')
                        elif filename.lower().endswith('.docx'):
                            attachment.add_header('Content-Type', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')
                        elif filename.lower().endswith('.pptx') or filename.lower().endswith('.ppt'):
                            attachment.add_header('Content-Type', 'application/vnd.ms-powerpoint')
                        
                        attachment.add_header('Content-Disposition', 'attachment', filename=filename)
                        msg.attach(attachment)
                        logger.info(f"  Attached: {filename} ({len(file_data)} bytes)")
                except Exception as e:
                    logger.error(f" Failed to attach {filepath}: {e}")
                    continue
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                
                recipients = [recipient]
                if cc:
                    recipients.extend(cc)
                
                server.send_message(msg)
            
            logger.info(f" Email sent to {recipient}")
            return True
        
        except Exception as e:
            logger.error(f" Email sending failed: {e}")
            return False
    
    def generate_outreach_email_with_reasoning(self, topic: str, 
                                              audience: str = "general public") -> str:
        """
        Generate outreach email using multi-step reasoning (LangChain-style).
        
        Step 1: Analyze audience and topic
        Step 2: Identify key messages
        Step 3: Generate email with appropriate tone
        
        Args:
            topic: Email topic
            audience: Target audience
        
        Returns:
            Email body text
        """
        if not self.llm_client:
            return self._fallback_email_body(topic, audience)
        
        try:
            # Step 1: Analyze audience and topic
            analyze_prompt = f"""Analyze the audience and topic for email outreach.

Topic: {topic}
Audience: {audience}

Provide analysis in JSON format:
{{
  "audience_characteristics": ["characteristic1", "characteristic2"],
  "key_interests": ["interest1", "interest2"],
  "appropriate_tone": "tone description",
  "main_message": "primary message to convey"
}}"""
            
            analysis_result = self.llm_client.generate(analyze_prompt, temperature=0.6, max_tokens=500)
            
            # Step 2: Identify key points
            keypoints_prompt = f"""Based on the analysis, identify 2-3 key points to include in the email.

Analysis: {analysis_result}
Topic: {topic}
Audience: {audience}

Provide key points in JSON format:
{{
  "key_points": [
    {{
      "point": "point text",
      "supporting_fact": "relevant fact or statistic"
    }}
  ],
  "call_to_action": "suggested call to action"
}}"""
            
            keypoints_result = self.llm_client.generate(keypoints_prompt, temperature=0.6, max_tokens=500)
            
            # Step 3: Generate email
            email_prompt = f"""Write a concise outreach email (150-250 words) based on the analysis and key points.

Analysis: {analysis_result}
Key Points: {keypoints_result}
Topic: {topic}
Audience: {audience}

Email requirements:
- Use the appropriate tone identified in analysis
- Include an engaging introduction
- Cover the key points naturally
- Include the call-to-action
- Keep it concise (150-250 words)
- Return only the email body text (no subject line, no headers)

Email body:"""
            
            email_body = self.llm_client.generate(email_prompt, temperature=0.7, max_tokens=800)
            
            # Clean up
            email_body = email_body.strip()
            if email_body.startswith("```"):
                lines = email_body.split("\n")
                email_body = "\n".join([l for l in lines if not l.startswith("```")])
            
            return email_body.strip()
        
        except Exception as e:
            logger.error(f"Email generation with reasoning failed: {e}")
            return self._fallback_email_body(topic, audience)
    
    def _fallback_email_body(self, topic: str, audience: str) -> str:
        """Fallback email body if generation fails."""
        if self.llm_client:
            try:
                prompt = f"""Write a concise outreach email (150-250 words) about the importance of sustainability 
regarding {topic} for the {audience}. 

Tone: clear, encouraging, and factual. Include an intro, 2-3 key points, and a brief call-to-action. 
Return only the email body text (no headers)."""
                
                return self.llm_client.generate(prompt, temperature=0.7, max_tokens=800).strip()
            except:
                pass
        
        return f"""Dear {audience},

We wanted to share important information about {topic} and its impact on environmental sustainability.

{topic.title()} is a critical area that affects our planet's future. Understanding and taking action on this topic can make a significant difference.

We encourage you to learn more and consider how you can contribute to positive change in this area.

Best regards,
Environmental Sustainability Team"""
    
    def send_quiz_result(self, recipient: str, quiz_filepath: str, 
                        topic: str, score: Optional[str] = None) -> bool:
        """
        Send quiz document via email with LangChain-generated body.
        
        Args:
            recipient: Recipient email
            quiz_filepath: Path to quiz document
            topic: Quiz topic
            score: Optional score/results
            
        Returns:
            Success status
        """
        subject = f"Environmental Sustainability Quiz: {topic.title()}"
        
        # Generate personalized email body using LangChain reasoning
        body = self.generate_outreach_email_with_reasoning(
            topic=f"quiz on {topic}",
            audience="student"
        )
        
        # Add score if provided
        if score:
            body = f"Your score: {score}\n\n" + body
        
        return self.send_document(recipient, subject, body, [quiz_filepath])
    
    def send_study_guide(self, recipient: str, guide_filepath: str, 
                        topic: str) -> bool:
        """Send study guide via email with LangChain-generated body."""
        subject = f"Study Guide: {topic.title()}"
        
        # Generate personalized email body
        body = self.generate_outreach_email_with_reasoning(
            topic=f"study guide on {topic}",
            audience="student"
        )
        
        return self.send_document(recipient, subject, body, [guide_filepath])
    
    def send_outreach_email(self, to_emails: List[str], topic: str, 
                           audience: str = "general public",
                           attachments: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate and send outreach email using LangChain multi-step reasoning.
        
        Args:
            to_emails: List of recipient email addresses
            topic: Email topic
            audience: Target audience
            attachments: Optional file attachments
        
        Returns:
            Dict with send results
        """
        body = self.generate_outreach_email_with_reasoning(topic=topic, audience=audience)
        subject = f"Sustainability Matters: {topic.title()}"
        
        results = []
        for recipient in to_emails:
            success = self.send_document(
                recipient=recipient,
                subject=subject,
                body=body,
                attachments=attachments or []
            )
            results.append({
                "recipient": recipient,
                "sent": success
            })
        
        return {
            "topic": topic,
            "audience": audience,
            "subject": subject,
            "body_preview": body[:200] + "..." if len(body) > 200 else body,
            "recipients": results,
            "total_sent": sum(1 for r in results if r["sent"])
        }
