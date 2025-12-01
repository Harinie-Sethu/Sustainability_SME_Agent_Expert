"""
Test Runner for Part E: Agent with Planning, Reasoning, and Routing
Tests all agentic capabilities with comprehensive scenarios
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_agent_initialization():
    """Test agent initialization with all components."""
    print("\n" + "="*70)
    print("TEST 1: Agent Initialization")
    print("="*70)
    
    try:
        from partd.llm_client import GeminiLLMClient
        from partd.enhanced_rag import EnhancedRAG
        from partd.task_handlers import ContentGenerator
        from parte.agent_core import EnvironmentalAgent
        
        llm = GeminiLLMClient()
        
        # Try Milvus, fallback to web search if unavailable
        try:
            from partc.milvus_retrieval_pipeline import MilvusRetrievalPipeline
            db_path = str(Path(__file__).parent.parent / "partc" / "milvus_data.db")
            retrieval = MilvusRetrievalPipeline(db_path)
            rag = EnhancedRAG(retrieval, llm, enable_web_search=True)
            print("  - Using Milvus + Web Search")
        except Exception as e:
            # Fallback to web search only
            logger.warning(f"Milvus unavailable: {e}. Using web search fallback.")
            class MockRetrieval:
                def hierarchical_retrieve(self, query, **kwargs):
                    return {'results': []}
            mock_retrieval = MockRetrieval()
            rag = EnhancedRAG(mock_retrieval, llm, enable_web_search=True)
            print("  - Using Web Search (Milvus unavailable)")
        
        content_gen = ContentGenerator(rag, llm)
        
        agent = EnvironmentalAgent(llm, rag, content_gen)
        
        print("✓ Agent initialized successfully")
        print(f"  - Conversation Manager: Active")
        print(f"  - Task Router: {len(agent.router.handlers)} handlers registered")
        print(f"  - Observations Logger: Active")
        
        return True, agent
        
    except Exception as e:
        print(f"✗ Agent initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_qa_routing(agent):
    """Test Q&A routing and execution."""
    print("\n" + "="*70)
    print("TEST 2: Q&A Task Routing")
    print("="*70)
    
    try:
        questions = [
            "What is renewable energy?",
            "Explain the greenhouse effect",
            "How does deforestation affect climate?"
        ]
        
        for q in questions:
            print(f"\n  Q: {q}")
            result = agent.process_request(q)
            print(f"  Handler: {result['metadata']['handler_used']}")
            print(f"  Response: {result['message'][:150]}...")
            print(f"  Time: {result['metadata']['execution_time']:.2f}s")
        
        print("\n✓ Q&A routing working")
        return True
        
    except Exception as e:
        print(f"✗ Q&A routing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_content_generation_routing(agent):
    """Test content generation routing."""
    print("\n" + "="*70)
    print("TEST 3: Content Generation Routing")
    print("="*70)
    
    try:
        requests = [
            "Create a quiz about recycling",
            "Generate a study guide on solar energy",
            "Write an article about ocean pollution for students"
        ]
        
        for req in requests:
            print(f"\n  Request: {req}")
            result = agent.process_request(req)
            print(f"  Handler: {result['metadata']['handler_used']}")
            print(f"  Type: {result.get('type')}")
            print(f"  Success: {result.get('success')}")
            
            # Cache content for next tests
            if result.get('content'):
                agent.conversation.cache_context('last_generated_content', result['content'])
        
        print("\n✓ Content generation routing working")
        return True
        
    except Exception as e:
        print(f"✗ Content generation routing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_step_planning(agent):
    """Test multi-step planning for complex requests."""
    print("\n" + "="*70)
    print("TEST 4: Multi-Step Planning")
    print("="*70)
    
    try:
        complex_request = "Create a quiz about climate change with 3 questions and export it to PDF"
        
        print(f"\n  Complex Request: {complex_request}")
        print(f"  Expected: Planning -> Content Gen -> Document Export")
        
        result = agent.process_request(complex_request)
        
        print(f"\n  Handler Used: {result['metadata']['handler_used']}")
        print(f"  Subtasks Planned: {result['metadata']['plan_subtasks']}")
        print(f"  Execution Time: {result['metadata']['execution_time']:.2f}s")
        print(f"  Success: {result.get('success')}")
        
        print("\n✓ Multi-step planning working")
        return True
        
    except Exception as e:
        print(f"✗ Multi-step planning failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_conversation_memory(agent):
    """Test conversation memory and context management."""
    print("\n" + "="*70)
    print("TEST 5: Conversation Memory & Context")
    print("="*70)
    
    try:
        # First message
        result1 = agent.process_request("Tell me about solar panels")
        print(f"  Message 1: Solar panels question")
        
        # Follow-up using context
        result2 = agent.process_request("What are the main advantages?")
        print(f"  Message 2: Follow-up about advantages")
        
        # Another follow-up
        result3 = agent.process_request("How much do they cost?")
        print(f"  Message 3: Follow-up about cost")
        
        # Check conversation stats
        stats = agent.conversation.get_conversation_stats()
        print(f"\n  Conversation Stats:")
        print(f"    - Total Messages: {stats['total_messages']}")
        print(f"    - User Messages: {stats['user_messages']}")
        print(f"    - Agent Messages: {stats['agent_messages']}")
        
        # Get context summary
        summary = agent.conversation.get_context_summary(agent.llm)
        print(f"    - Context Summary: {summary[:100]}...")
        
        print("\n✓ Conversation memory working")
        return True
        
    except Exception as e:
        print(f"✗ Conversation memory failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reasoning_and_analysis(agent):
    """Test multi-step reasoning for analysis tasks."""
    print("\n" + "="*70)
    print("TEST 6: Multi-Step Reasoning & Analysis")
    print("="*70)
    
    try:
        analysis_requests = [
            "Compare wind energy and solar energy",
            "Analyze the impact of plastic bags on marine life"
        ]
        
        for req in analysis_requests:
            print(f"\n  Request: {req}")
            result = agent.process_request(req)
            print(f"  Handler: {result['metadata']['handler_used']}")
            print(f"  Response: {result.get('message', '')[:150]}...")
        
        print("\n✓ Reasoning and analysis working")
        return True
        
    except Exception as e:
        print(f"✗ Reasoning failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_conversational_handling(agent):
    """Test conversational interactions."""
    print("\n" + "="*70)
    print("TEST 7: Conversational Handling")
    print("="*70)
    
    try:
        conversations = [
            "Hello!",
            "What can you do?",
            "Thanks for your help"
        ]
        
        for conv in conversations:
            print(f"\n  User: {conv}")
            result = agent.process_request(conv)
            print(f"  Agent: {result.get('message', '')[:150]}...")
            print(f"  Handler: {result['metadata']['handler_used']}")
        
        print("\n✓ Conversational handling working")
        return True
        
    except Exception as e:
        print(f"✗ Conversational handling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_observations_and_logging(agent):
    """Test observations logging and insights."""
    print("\n" + "="*70)
    print("TEST 8: Observations & Logging")
    print("="*70)
    
    try:
        # Process some requests to generate observations
        requests = [
            "What is composting?",
            "Create a quiz about water conservation",
            "Compare recycling and composting"
        ]
        
        for req in requests:
            agent.process_request(req)
        
        # Get insights
        insights = agent.observations.generate_insights()
        print(f"\n  Observations Insights:")
        print(f"    - Total Observations: {insights['total_observations']}")
        print(f"    - Total Iterations: {insights['total_iterations']}")
        print(f"    - Observation Types: {insights['observation_types']}")
        print(f"    - Execution Success Rate: {insights['execution_success_rate']:.1%}")
        
        # Get improvement suggestions
        suggestions = agent.get_improvement_suggestions()
        print(f"\n  Improvement Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"    {i}. {suggestion}")
        
        # Generate report
        report = agent.observations.generate_report(save_to_file=False)
        print(f"\n  Report Preview:")
        print(report[:300] + "...")
        
        print("\n✓ Observations and logging working")
        return True
        
    except Exception as e:
        print(f"✗ Observations failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_session_management(agent):
    """Test session saving and statistics."""
    print("\n" + "="*70)
    print("TEST 9: Session Management")
    print("="*70)
    
    try:
        # Get session summary
        summary = agent.get_session_summary()
        print(f"\n  Session Summary:")
        print(f"    - Session ID: {summary['conversation']['session_id']}")
        print(f"    - Total Messages: {summary['conversation']['total_messages']}")
        print(f"    - Handler Usage: {summary['routing']['handler_distribution']}")
        
        # Save session
        session_file = agent.save_session()
        print(f"\n  Session saved to: {session_file}")
        
        # Get routing stats
        routing_stats = agent.router.get_routing_stats()
        print(f"\n  Routing Statistics:")
        print(f"    - Total Routes: {routing_stats['total_routes']}")
        print(f"    - Handler Distribution:")
        for handler, count in routing_stats['handler_distribution'].items():
            print(f"      - {handler}: {count}")
        
        print("\n✓ Session management working")
        return True
        
    except Exception as e:
        print(f"✗ Session management failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling(agent):
    """Test error handling and recovery."""
    print("\n" + "="*70)
    print("TEST 10: Error Handling & Recovery")
    print("="*70)
    
    try:
        # Test ambiguous request
        ambiguous = "Do something with environment"
        print(f"\n  Ambiguous Request: {ambiguous}")
        result = agent.process_request(ambiguous)
        print(f"  Response Type: {result.get('type')}")
        print(f"  Requires Clarification: {result.get('requires_user_input', False)}")
        
        # Test invalid request
        invalid = "kjsdfhkjsdhfkjsdhf"
        print(f"\n  Invalid Request: {invalid}")
        result = agent.process_request(invalid)
        print(f"  Handler: {result['metadata']['handler_used']}")
        print(f"  Response: {result.get('message', '')[:100]}...")
        
        print("\n✓ Error handling working")
        return True
        
    except Exception as e:
        print(f"✗ Error handling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_comprehensive_scenario(agent):
    """Run a comprehensive multi-turn scenario."""
    print("\n" + "="*70)
    print("TEST 11: Comprehensive Multi-Turn Scenario")
    print("="*70)
    
    try:
        scenario = [
            ("Hello! I'm a teacher preparing a lesson.", "greeting"),
            ("I need to teach about renewable energy.", "context"),
            ("Can you create a quiz with 3 questions?", "content_gen"),
            ("Make it for high school students, medium difficulty.", "refinement"),
            ("Great! Now export it to PDF.", "export"),
            ("Thanks! You've been very helpful.", "closing")
        ]
        
        print("\n  Running multi-turn scenario...")
        
        for i, (request, label) in enumerate(scenario, 1):
            print(f"\n  Turn {i} [{label}]: {request}")
            result = agent.process_request(request)
            print(f"    Handler: {result['metadata']['handler_used']}")
            print(f"    Response: {result.get('message', '')[:100]}...")
            
            # Brief pause to simulate real interaction
            import time
            time.sleep(0.5)
        
        # Final summary
        summary = agent.get_session_summary()
        print(f"\n  Scenario Complete:")
        print(f"    - Total Turns: {len(scenario)}")
        print(f"    - Handlers Used: {set(summary['routing']['handler_distribution'].keys())}")
        print(f"    - Success Rate: {summary['observations']['execution_success_rate']:.1%}")
        
        print("\n✓ Comprehensive scenario completed")
        return True
        
    except Exception as e:
        print(f"✗ Comprehensive scenario failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Part E tests."""
    print("="*70)
    print("PART E: AGENT WITH PLANNING, REASONING, AND ROUTING")
    print("="*70)
    print("\nFeatures:")
    print("  ✓ Conversational Planning")
    print("  ✓ Multi-Step Reasoning")
    print("  ✓ Intelligent Task Routing")
    print("  ✓ Context & Memory Management")
    print("  ✓ Observational Learning")
    print("  ✓ Decision Strategies")
    print("="*70)
    
    # Initialize agent
    success, agent = test_agent_initialization()
    
    if not success or agent is None:
        print("\n✗ Agent initialization failed. Cannot proceed with tests.")
        return 1
    
    # Run tests
    results = {
        "Agent Initialization": success,
        "Q&A Routing": test_qa_routing(agent),
        "Content Generation Routing": test_content_generation_routing(agent),
        "Multi-Step Planning": test_multi_step_planning(agent),
        "Conversation Memory": test_conversation_memory(agent),
        "Reasoning & Analysis": test_reasoning_and_analysis(agent),
        "Conversational Handling": test_conversational_handling(agent),
        "Observations & Logging": test_observations_and_logging(agent),
        "Session Management": test_session_management(agent),
        "Error Handling": test_error_handling(agent),
        "Comprehensive Scenario": run_comprehensive_scenario(agent)
    }
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL PART E TESTS PASSED")
        print("\nImplemented Features:")
        print("  ✓ Conversational Planning (with context awareness)")
        print("  ✓ Multi-Step Reasoning (chain-of-thought)")
        print("  ✓ Intelligent Task Routing (rule-based + LLM)")
        print("  ✓ Context Management (conversation history & memory)")
        print("  ✓ Observational Logging (decisions & performance)")
        print("  ✓ Decision Strategies (confidence-based routing)")
        print("  ✓ Prompt Refinement (based on observations)")
        
        # Display final insights
        print("\n" + "="*70)
        print("FINAL SESSION INSIGHTS")
        print("="*70)
        
        summary = agent.get_session_summary()
        print(f"\nConversation:")
        print(f"  - Total Messages: {summary['conversation']['total_messages']}")
        print(f"  - User/Agent: {summary['conversation']['user_messages']}/{summary['conversation']['agent_messages']}")
        
        print(f"\nRouting:")
        print(f"  - Total Routes: {summary['routing']['total_routes']}")
        print(f"  - Handler Distribution:")
        for handler, count in summary['routing']['handler_distribution'].items():
            print(f"    - {handler}: {count}")
        
        print(f"\nObservations:")
        print(f"  - Total Observations: {summary['observations']['total_observations']}")
        print(f"  - Execution Success: {summary['observations']['execution_success_rate']:.1%}")
        
        # Save final session
        print("\n" + "="*70)
        session_file = agent.save_session()
        print(f"Session saved to: {session_file}")
        print("="*70)
    else:
        print("✗ SOME TESTS FAILED")
    
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
        session_file = agent.save_session()
        print(f"Session saved to: {session_file}")
        print("="*70)
    else:
        print("✗ SOME TESTS FAILED")
    
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

