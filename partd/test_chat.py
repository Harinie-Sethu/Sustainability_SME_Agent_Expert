#!/usr/bin/env python3
"""
Interactive chat test for Part D
Tests Q&A with conversation history
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import os
os.environ['GEMINI_API_KEY'] = 'AIzaSyDcmWDDD9auO6kyPYvMQbgMZRfZq02ydjo'

from partd.llm_client import GeminiLLMClient
from partd.enhanced_rag import EnhancedRAG
from partd.task_handlers import ContentGenerator

# Create mock retrieval (uses web search)
class MockRetrieval:
    def hierarchical_retrieve(self, query, **kwargs):
        return {'results': []}

print("="*70)
print("PART D - INTERACTIVE CHAT TEST")
print("="*70)
print("Type your questions about sustainability and environment.")
print("Type 'quit', 'exit', or 'q' to exit.\n")

llm = GeminiLLMClient()
mock_retrieval = MockRetrieval()
rag = EnhancedRAG(mock_retrieval, llm, enable_web_search=True)
generator = ContentGenerator(rag, llm)

conversation_history = []

while True:
    question = input("You: ")
    if question.lower() in ['quit', 'exit', 'q']:
        print("\nGoodbye!")
        break
    
    if not question.strip():
        continue
    
    print("\nThinking...")
    try:
        result = generator.answer_question(question, conversation_history=conversation_history)
        print(f"\nAssistant: {result['answer']}\n")
        print(f"[Sources: {result['metadata']['num_web_sources']} web, "
              f"Confidence: {result['metadata']['retrieval_confidence']:.3f}]\n")
        
        conversation_history.append({'role': 'user', 'content': question})
        conversation_history.append({'role': 'assistant', 'content': result['answer']})
    except Exception as e:
        print(f"\nError: {e}\n")

