"""
Test runner for Part D: Core SME Capabilities
Tests all features including BONUS self-learning
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_llm_client():
    """Test LLM client with chain generation."""
    print("\n" + "="*70)
    print("TEST 1: LLM Client with Chain Generation")
    print("="*70)
    
    try:
        from partd.llm_client import GeminiLLMClient
        
        client = GeminiLLMClient()
        
        # Test simple generation
        response = client.generate("What is sustainability?")
        print(f" Simple generation works: {response[:100]}...")
        
        # Test chain generation
        chain_steps = [
            {
                "name": "step1",
                "prompt": "List 3 environmental topics in JSON format: {\"topics\": [\"t1\", \"t2\", \"t3\"]}",
                "output_key": "topics"
            },
            {
                "name": "step2",
                "prompt": "Based on these topics: {topics}\n\nChoose the most important one and explain why in one sentence.",
                "output_key": "analysis"
            }
        ]
        
        result = client.chain_generate(chain_steps)
        print(f" Chain generation works:")
        print(f"  Step 1 output: {result.get('topics', '')[:100]}...")
        print(f"  Step 2 output: {result.get('analysis', '')[:100]}...")
        
        return True
        
    except Exception as e:
        print(f" LLM client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_with_web_search():
    """Test RAG with web search fallback."""
    print("\n" + "="*70)
    print("TEST 2: Enhanced RAG with Web Search")
    print("="*70)
    
    try:
        from partd.llm_client import GeminiLLMClient
        from partc.milvus_retrieval_pipeline import MilvusRetrievalPipeline
        from partd.enhanced_rag import EnhancedRAG
        
        db_path = str(Path(__file__).parent.parent / "partc" / "milvus_data.db")
        llm = GeminiLLMClient()
        retrieval = MilvusRetrievalPipeline(db_path)
        rag = EnhancedRAG(retrieval, llm)
        
        # Test with local knowledge
        result1 = rag.answer_question("What causes deforestation?")
        print(f" RAG works")
        print(f"  Answer: {result1['answer'][:150]}...")
        print(f"  Used web search: {result1['metadata']['used_web_search']}")
        print(f"  Confidence: {result1['metadata']['retrieval_confidence']:.3f}")
        
        return True
        
    except Exception as e:
        print(f" RAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_task1_qa():
    """Test Task 1: Expert Q&A."""
    print("\n" + "="*70)
    print("TEST 3: TASK 1 - Expert Q&A")
    print("="*70)
    
    try:
        from partd.llm_client import GeminiLLMClient
        from partc.milvus_retrieval_pipeline import MilvusRetrievalPipeline
        from partd.enhanced_rag import EnhancedRAG
        from partd.task_handlers import ContentGenerator
        
        db_path = str(Path(__file__).parent.parent / "partc" / "milvus_data.db")
        llm = GeminiLLMClient()
        retrieval = MilvusRetrievalPipeline(db_path)
        rag = EnhancedRAG(retrieval, llm)
        generator = ContentGenerator(rag, llm)
        
        questions = [
            "What is ocean acidification?",
            "How does plastic pollution affect marine life?",
        ]
        
        for q in questions:
            print(f"\n  Q: {q}")
            result = generator.answer_question(q)
            print(f"  A: {result['answer'][:200]}...")
            print(f"  Sources: {result['metadata']['num_local_sources']} local")
        
        print(f"\n Task 1 (Expert Q&A) working")
        return True
        
    except Exception as e:
        print(f" Task 1 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_task2_quiz():
    """Test Task 2: Quiz Generation with multi-step reasoning."""
    print("\n" + "="*70)
    print("TEST 4: TASK 2 - Quiz Generation (Multi-Step)")
    print("="*70)
    
    try:
        from partd.llm_client import GeminiLLMClient
        from partc.milvus_retrieval_pipeline import MilvusRetrievalPipeline
        from partd.enhanced_rag import EnhancedRAG
        from partd.task_handlers import ContentGenerator
        
        db_path = str(Path(__file__).parent.parent / "partc" / "milvus_data.db")
        llm = GeminiLLMClient()
        retrieval = MilvusRetrievalPipeline(db_path)
        rag = EnhancedRAG(retrieval, llm)
        generator = ContentGenerator(rag, llm)
        
        print("\n  Generating quiz with multi-step reasoning...")
        quiz = generator.generate_quiz("recycling", num_questions=3, difficulty="easy")
        
        print(f" Quiz generated")
        print(f"  Topic: {quiz['topic']}")
        print(f"  Questions: {quiz['num_questions']}")
        print(f"  Used chain reasoning: {quiz['metadata'].get('used_chain_reasoning', False)}")
        
        if quiz.get('questions'):
            q = quiz['questions'][0]
            print(f"\n  Sample question: {q['question']}")
            print(f"  Answer: {q['correct_answer']}")
        
        return True
        
    except Exception as e:
        print(f" Quiz generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_task2_study_guide():
    """Test Task 2: Study Guide Generation."""
    print("\n" + "="*70)
    print("TEST 5: TASK 2 - Study Guide Generation")
    print("="*70)
    
    try:
        from partd.llm_client import GeminiLLMClient
        from partc.milvus_retrieval_pipeline import MilvusRetrievalPipeline
        from partd.enhanced_rag import EnhancedRAG
        from partd.task_handlers import ContentGenerator
        
        db_path = str(Path(__file__).parent.parent / "partc" / "milvus_data.db")
        llm = GeminiLLMClient()
        retrieval = MilvusRetrievalPipeline(db_path)
        rag = EnhancedRAG(retrieval, llm)
        generator = ContentGenerator(rag, llm)
        
        print("\n  Generating study guide...")
        guide = generator.generate_study_guide("renewable energy")
        
        print(f" Study guide generated")
        print(f"  Topic: {guide['topic']}")
        print(f"  Content length: {len(guide['content'])} characters")
        print(f"  Preview: {guide['content'][:150]}...")
        
        return True
        
    except Exception as e:
        print(f" Study guide test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_task2_awareness():
    """Test Task 2: Awareness Material Generation."""
    print("\n" + "="*70)
    print("TEST 6: TASK 2 - Awareness Materials")
    print("="*70)
    
    try:
        from partd.llm_client import GeminiLLMClient
        from partc.milvus_retrieval_pipeline import MilvusRetrievalPipeline
        from partd.enhanced_rag import EnhancedRAG
        from partd.task_handlers import ContentGenerator
        
        db_path = str(Path(__file__).parent.parent / "partc" / "milvus_data.db")
        llm = GeminiLLMClient()
        retrieval = MilvusRetrievalPipeline(db_path)
        rag = EnhancedRAG(retrieval, llm)
        generator = ContentGenerator(rag, llm)
        
        formats = ["article", "social_media", "poster_text"]
        
        for fmt in formats:
            print(f"\n  Generating {fmt}...")
            material = generator.generate_awareness_material("plastic pollution", fmt, "students")
            print(f"   {fmt} generated: {len(material['content'])} characters")
        
        print(f"\n Awareness materials working")
        return True
        
    except Exception as e:
        print(f" Awareness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_document_export():
    """Test document export functionality."""
    print("\n" + "="*70)
    print("TEST 7: Document Export (PDF/DOCX)")
    print("="*70)
    
    try:
        from partd.llm_client import GeminiLLMClient
        from partc.milvus_retrieval_pipeline import MilvusRetrievalPipeline
        from partd.enhanced_rag import EnhancedRAG
        from partd.task_handlers import ContentGenerator
        from partd.export_tools import DocumentExporter
        
        db_path = str(Path(__file__).parent.parent / "partc" / "milvus_data.db")
        llm = GeminiLLMClient()
        retrieval = MilvusRetrievalPipeline(db_path)
        rag = EnhancedRAG(retrieval, llm)
        generator = ContentGenerator(rag, llm)
        exporter = DocumentExporter()
        
        # Generate quiz
        quiz = generator.generate_quiz("water conservation", num_questions=2)
        
        # Export to DOCX
        docx_path = exporter.export_quiz_to_docx(quiz)
        print(f" DOCX export: {docx_path}")
        
        # Export to PDF
        pdf_path = exporter.export_quiz_to_pdf(quiz)
        print(f" PDF export: {pdf_path}")
        
        return True
        
    except Exception as e:
        print(f" Export test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_self_learning():
    """Test BONUS: Self-learning from feedback."""
    print("\n" + "="*70)
    print("TEST 8: BONUS - Self-Learning from Feedback")
    print("="*70)
    
    try:
        from partd.llm_client import GeminiLLMClient
        from partc.milvus_retrieval_pipeline import MilvusRetrievalPipeline
        from partd.enhanced_rag import EnhancedRAG
        from partd.task_handlers import ContentGenerator
        
        db_path = str(Path(__file__).parent.parent / "partc" / "milvus_data.db")
        llm = GeminiLLMClient()
        retrieval = MilvusRetrievalPipeline(db_path)
        rag = EnhancedRAG(retrieval, llm, feedback_storage="partd/test_feedback")
        generator = ContentGenerator(rag, llm)
        
        # Answer a question
        result = generator.answer_question("What is composting?")
        
        # Provide feedback
        print("\n  Recording feedback...")
        generator.provide_feedback(
            question="What is composting?",
            answer=result['answer'],
            feedback_type="positive",
            feedback_text="Great explanation with good examples",
            rating=5
        )
        
        generator.provide_feedback(
            question="What is composting?",
            answer=result['answer'],
            feedback_type="suggestion",
            feedback_text="Could use more detail about the process",
            rating=4
        )
        
        # Get insights
        insights = generator.get_learning_insights()
        print(f" Self-learning working")
        print(f"  Total feedback: {insights.get('total_feedback', 0)}")
        print(f"  Positive: {insights.get('positive_feedback', 0)}")
        print(f"  Average rating: {insights.get('average_rating', 0):.2f}")
        
        # Answer another question with feedback context
        result2 = generator.answer_question("Explain composting in detail")
        print(f"  Applied feedback: {result2['metadata'].get('applied_feedback', False)}")
        
        return True
        
    except Exception as e:
        print(f" Self-learning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_email_sending():
    """Test email functionality with LangChain reasoning."""
    print("\n" + "="*70)
    print("TEST 9: Email Document Delivery (LangChain Multi-Step)")
    print("="*70)
    
    try:
        from partd.llm_client import GeminiLLMClient
        from partc.milvus_retrieval_pipeline import MilvusRetrievalPipeline
        from partd.enhanced_rag import EnhancedRAG
        from partd.task_handlers import ContentGenerator
        from partd.export_tools import DocumentExporter
        
        db_path = str(Path(__file__).parent.parent / "partc" / "milvus_data.db")
        llm = GeminiLLMClient()
        retrieval = MilvusRetrievalPipeline(db_path)
        rag = EnhancedRAG(retrieval, llm)
        generator = ContentGenerator(rag, llm)
        exporter = DocumentExporter()
        
        # Generate quiz
        print("\n  Generating quiz document...")
        quiz = generator.generate_quiz("climate change", num_questions=2)
        docx_path = exporter.export_quiz_to_docx(quiz)
        print(f"   Generated: {docx_path}")
        
        # Test email generation (without actually sending)
        print("\n  Testing email generation with LangChain multi-step reasoning...")
        from partd.email_tool import EmailTool
        email_tool = EmailTool(llm_client=llm)
        
        email_body = email_tool.generate_outreach_email_with_reasoning(
            topic="climate change quiz",
            audience="students"
        )
        print(f"   Generated email body ({len(email_body)} chars)")
        print(f"  Preview: {email_body[:150]}...")
        
        # Test outreach email generation
        print("\n  Testing outreach email generation...")
        result = generator.send_outreach_email(
            to_emails=["reshmakreveendran629@gmail.com", "hariniesethu@gmail.com"],
            topic="renewable energy",
            audience="students"
        )
        print(f"   Outreach email generated")
        print(f"  Subject: {result['subject']}")
        print(f"  Body preview: {result['body_preview']}")
        
        print(f"\n   To actually send emails, set environment variables:")
        print(f"     export SENDER_EMAIL='your_email@gmail.com'")
        print(f"     export SENDER_PASSWORD='your_app_password'")
        
        return True
        
    except Exception as e:
        print(f" Email test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Part D tests."""
    print("="*70)
    print("PART D: CORE SME CAPABILITIES - COMPLETE TEST SUITE")
    print("="*70)
    print("\nFeatures:")
    print("   Task 1: Expert Q&A with RAG + Web Search")
    print("   Task 2: Educational Content Generation")
    print("    - Quiz with multi-step reasoning")
    print("    - Study Guides")
    print("    - Awareness Materials")
    print("   Document Export (PDF/DOCX)")
    print("   BONUS: Self-Learning from Feedback")
    print("="*70)
    
    '''results = {
        "LLM Client + Chain": test_llm_client(),
        "RAG + Web Search": test_rag_with_web_search(),
        "Task 1: Q&A": test_task1_qa(),
        "Task 2: Quiz": test_task2_quiz(),
        "Task 2: Study Guide": test_task2_study_guide(),
        "Task 2: Awareness": test_task2_awareness(),
        "Document Export": test_document_export(),
        "BONUS: Self-Learning": test_self_learning()
    }'''
    results = {
        "LLM Client + Chain": test_llm_client(),
        "RAG + Web Search": test_rag_with_web_search(),
        "Task 1: Q&A": test_task1_qa(),
        "Task 2: Quiz": test_task2_quiz(),
        "Task 2: Study Guide": test_task2_study_guide(),
        "Task 2: Awareness": test_task2_awareness(),
        "Document Export": test_document_export(),
        "BONUS: Self-Learning": test_self_learning(),
        "Email Delivery (LangChain)": test_email_sending()
    }
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = " PASSED" if passed else " FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    '''print("\n" + "="*70)
    if all_passed:
        print(" ALL PART D TESTS PASSED")
        print("\nImplemented Features:")
        print("   Task 1: Expert Q&A (RAG-based)")
        print("   Task 2: Content Generation (Quiz, Study Guide, Awareness)")
        print("   Multi-Step Reasoning (Chain generation)")
        print("   Adaptive Explanations (Context-aware)")
        print("   Document Export (PDF/DOCX)")
        print("   BONUS: Self-Learning from Feedback")
    else:
        print(" SOME TESTS FAILED")
    
    print("="*70)'''
    print("="*70)
    if all_passed:
        print(" ALL PART D TESTS PASSED")
        print("\nImplemented Features:")
        print("   Task 1: Expert Q&A (RAG-based)")
        print("   Task 2: Content Generation (Quiz, Study Guide, Awareness)")
        print("   Multi-Step Reasoning (Chain generation)")
        print("   Adaptive Explanations (Context-aware)")
        print("   Document Export (PDF/DOCX)")
        print("   Email Delivery (LangChain multi-step reasoning)")
        print("   BONUS: Self-Learning from Feedback")
    else:
        print(" SOME TESTS FAILED")
    
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
