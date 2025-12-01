"""
Enhanced RAG System with Web Search Fallback
Connects retrieval with generation and includes self-learning capabilities
"""

import importlib
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, List, Optional
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedRAG:
    """
    RAG system with:
    - Local vector database retrieval
    - Web search fallback
    - Self-learning from feedback (BONUS)
    """
    
    def __init__(self, retrieval_pipeline, llm_client, 
                 confidence_threshold: float = 0.3,  # Lowered from 0.5 to be less strict
                 enable_web_search: bool = True,
                 feedback_storage: Optional[str] = None):
        """
        Initialize Enhanced RAG.
        
        Args:
            retrieval_pipeline: Milvus retrieval pipeline
            llm_client: Gemini LLM client
            confidence_threshold: Minimum score to trust local results
            enable_web_search: Enable web search fallback
            feedback_storage: Path to store feedback for self-learning (BONUS)
        """
        self.retrieval = retrieval_pipeline
        self.llm = llm_client
        self.confidence_threshold = confidence_threshold
        self.enable_web_search = enable_web_search
        self.ddgs_class = None
        
        # BONUS: Self-learning storage
        self.feedback_storage = Path(feedback_storage) if feedback_storage else Path("partd/feedback_storage")
        self.feedback_storage.mkdir(exist_ok=True, parents=True)
        
        # Initialize web search
        self._init_web_search_backend()

    def _init_web_search_backend(self):
        """Initialize DuckDuckGo/ DDGS web search backend."""
        if not self.enable_web_search:
            self.web_search_available = False
            self.ddgs_class = None
            logger.info("Web search disabled via configuration")
            return
        
        backends = [
            ("ddgs", "✓ Web search enabled (ddgs)"),
            ("duckduckgo_search", "✓ Web search enabled (duckduckgo_search)")
        ]
        
        for module_name, success_msg in backends:
            try:
                module = importlib.import_module(module_name)
                self.ddgs_class = getattr(module, "DDGS")
                self.web_search_available = True
                logger.info(success_msg)
                return
            except ImportError:
                continue
        
        self.web_search_available = False
        self.ddgs_class = None
        logger.warning("⚠ Web search backend not available. Install 'ddgs' (preferred) or 'duckduckgo_search'.")
    
    def answer_question(self, question: str, top_k: int = 5, 
                       conversation_history: Optional[List[Dict]] = None) -> Dict:
        """
        Answer question using RAG with adaptive explanations.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            conversation_history: Previous conversation for context
            
        Returns:
            Dict with answer, sources, and metadata
        """
        logger.info(f"Processing question: {question}")
        
        # Step 1: Local retrieval
        local_results = self.retrieval.hierarchical_retrieve(
            question,
            top_k=top_k,
            expand_context=True,
            rerank=True
        )
        
        has_good_results = self._evaluate_retrieval_quality(local_results)
        
        # Filter library results by confidence threshold
        library_candidates = local_results.get('results', []) or []
        filtered_library = [
            result for result in library_candidates
            if result.get('rerank_score', result.get('score', 0)) >= self.confidence_threshold
        ]
        if filtered_library:
            logger.info(f"✓ {len(filtered_library)} library result(s) met cutoff ({self.confidence_threshold})")
        else:
            logger.info("⚠ No library results met the similarity cutoff")
        
        library_results = filtered_library[:top_k]
        num_needed = max(0, top_k - len(library_results))
        
        # Step 2: Web search if needed (fill remaining slots)
        web_results = []
        use_web_search = (
            (num_needed > 0 or not has_good_results) and
            self.enable_web_search and
            self.web_search_available
        )
        
        if use_web_search:
            if num_needed <= 0:
                # Ensure at least some web context if confidence is low
                num_needed = top_k
            logger.info(f"Fetching web results to fill {num_needed} slot(s)")
            web_results = self._perform_web_search(question, num_results=max(num_needed * 2, 5))[:num_needed]
        
        # Fallback if we still have no sources at all
        if not library_results and not web_results and library_candidates:
            logger.warning("No sources available after applying cutoff/web search. Falling back to top library results without cutoff.")
            library_results = library_candidates[:top_k]
        
        # Step 3: Build context using the filtered results plus web fillers
        context = self._build_context(
            library_results,
            web_results
        )
        
        # Step 4: Check feedback history for improvements (BONUS)
        feedback_context = self._get_feedback_context(question)
        
        # Step 5: Generate adaptive answer
        answer = self._generate_adaptive_answer(
            question, 
            context, 
            conversation_history,
            feedback_context,
            has_good_results, 
            use_web_search
        )
        
        return {
            "question": question,
            "answer": answer,
            "sources": {
                "local": library_results,
                "web": web_results
            },
            "metadata": {
                "used_web_search": len(web_results) > 0,
                "num_local_sources": len(library_results),
                "num_web_sources": len(web_results),
                "retrieval_confidence": self._calculate_confidence(local_results),
                "applied_feedback": len(feedback_context) > 0,
                "similarity_cutoff": self.confidence_threshold
            }
        }
    
    def _evaluate_retrieval_quality(self, results: Dict) -> bool:
        """Evaluate retrieval quality."""
        # Check if Milvus was available
        metadata = results.get('metadata', {})
        if not metadata.get('milvus_available', True):
            logger.info("Milvus not available - will use web search")
            return False
        
        if not results.get('results'):
            logger.info("No local results found - will use web search")
            return False
        
        top_result = results['results'][0]
        top_score = top_result.get('rerank_score', top_result.get('score', 0))
        
        logger.info(f"Top result score from library: {top_score:.3f} (threshold: {self.confidence_threshold})")
        logger.info(f"Source: {top_result.get('source_file', 'unknown')}")
        logger.info(f"Text preview: {top_result.get('text', '')[:100]}...")
        
        has_good = top_score >= self.confidence_threshold
        if has_good:
            logger.info("✓ Using library documents (good confidence)")
        else:
            logger.info("⚠ Library results below threshold - will supplement with web search")
        
        return has_good
    
    def _calculate_confidence(self, results: Dict) -> float:
        """Calculate confidence score."""
        if not results.get('results'):
            return 0.0
        
        top_result = results['results'][0]
        return float(top_result.get('rerank_score', top_result.get('score', 0)))
    
    def _perform_web_search(self, query: str, num_results: int = 3) -> List[Dict]:
        """Perform web search with proper query cleaning."""
        if not self.web_search_available or not self.ddgs_class:
            logger.warning("Web search backend unavailable")
            return []
        
        try:
            # Clean and prepare the search query
            # Remove any extra whitespace, special characters that might confuse the search
            cleaned_query = query.strip()
            
            # Remove common prefixes/suffixes that might interfere
            cleaned_query = cleaned_query.replace("what is", "").replace("what are", "").replace("tell me about", "")
            cleaned_query = cleaned_query.replace("explain", "").replace("describe", "")
            cleaned_query = cleaned_query.strip()
            
            # If query is too short after cleaning, use original
            if len(cleaned_query) < 3:
                cleaned_query = query.strip()
            
            # Log the actual query being used
            logger.info(f"Performing web search with query: '{cleaned_query}' (original: '{query}')")
            
            ddgs = self.ddgs_class()
            results = []
            
            # Perform the search
            try:
                # Use text search with proper parameters
                logger.debug(f"Calling ddgs.text with query: '{cleaned_query}', max_results: {num_results}")
                search_results = list(ddgs.text(cleaned_query, max_results=num_results))
                
                logger.debug(f"DuckDuckGo returned {len(search_results)} raw results")
                
                # Process and validate results
                for idx, result in enumerate(search_results):
                    title = result.get("title", "")
                    snippet = result.get("body", "")
                    url = result.get("href", "")
                    
                    # Log each result for debugging
                    logger.debug(f"Result {idx+1}: {title[:60]}... | URL: {url[:60]}")
                    
                    # Validate that results are relevant (basic check)
                    # If title/snippet don't contain any words from query, might be irrelevant
                    query_words = set(cleaned_query.lower().split())
                    # Filter out very short words (1-2 chars) and common stop words
                    query_words = {w for w in query_words if len(w) > 2 and w not in ['the', 'is', 'are', 'a', 'an', 'and', 'or', 'but']}
                    
                    if query_words:
                        result_text = (title + " " + snippet).lower()
                        result_words = set(result_text.split())
                        
                        # Check if at least one significant query word appears in result
                        matching_words = [w for w in query_words if w in result_words]
                        if not matching_words:
                            logger.warning(f"Skipping irrelevant result (no query words found): {title[:50]}...")
                            logger.debug(f"  Query words: {query_words}, Result text preview: {result_text[:100]}")
                            continue
                        else:
                            logger.debug(f"  Relevant result (matched words: {matching_words})")
                    
                    # Additional validation: check if URL looks suspicious or unrelated
                    if url and ('gmail' in url.lower() or 'google' in url.lower()) and not any(w in cleaned_query.lower() for w in ['gmail', 'google', 'email']):
                        logger.warning(f"Skipping potentially unrelated Google/Gmail result: {title[:50]}...")
                        continue
                    
                    results.append({
                        "title": title,
                        "snippet": snippet,
                        "url": url,
                        "source": "web_search"
                    })
                    
                    if len(results) >= num_results:
                        break
                
                # If we got fewer results than requested, log a warning
                if len(results) < num_results:
                    logger.warning(f"Only got {len(results)} relevant results out of {len(search_results)} raw results for query: '{cleaned_query}'")
                
                logger.info(f"✓ Web search returned {len(results)} relevant results for query: '{cleaned_query}'")
                
                # Log first result title for debugging
                if results:
                    logger.info(f"  First result: {results[0].get('title', 'No title')[:80]}")
                
            except Exception as search_error:
                logger.error(f"DuckDuckGo search error: {search_error}")
                # Try with original query if cleaned query fails
                if cleaned_query != query.strip():
                    logger.info(f"Retrying with original query: '{query}'")
                    try:
                        search_results = ddgs.text(query.strip(), max_results=num_results)
                        for result in search_results:
                            results.append({
                                "title": result.get("title", ""),
                                "snippet": result.get("body", ""),
                                "url": result.get("href", ""),
                                "source": "web_search"
                            })
                            if len(results) >= num_results:
                                break
                    except Exception as e2:
                        logger.error(f"Retry also failed: {e2}")
            
            return results
        
        except Exception as e:
            logger.error(f"Web search error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _build_context(self, local_results: List[Dict], web_results: List[Dict]) -> str:
        """Build context from retrieved sources."""
        context_parts = []
        
        # Add summary at the top
        context_parts.append("=== SOURCE SUMMARY ===\n")
        context_parts.append(f"Total sources provided: {len(local_results)} library source(s) and {len(web_results)} web source(s).\n")
        context_parts.append("Library sources are numbered 1 to {}. Web sources are numbered 1 to {}.\n".format(len(local_results), len(web_results)))
        context_parts.append("Do NOT mix the numbering - library and web sources have separate numbering systems.\n\n")
        
        if local_results:
            context_parts.append("=== LIBRARY SOURCES (from local knowledge base) ===\n")
            context_parts.append("These are documents from the local knowledge library. Reference them as 'Library Source X'.\n\n")
            for i, result in enumerate(local_results, 1):
                source = result.get('source_file', 'Unknown')
                score = result.get('rerank_score', result.get('score', 0))
                context_parts.append(f"[Library Source {i}: {source}] (Relevance Score: {score:.2f})")
                context_parts.append(result['text'][:1000])
                context_parts.append("")
        
        if web_results:
            context_parts.append("\n=== WEB SOURCES (from internet search) ===\n")
            context_parts.append("These are results from web search. Reference them as 'Web Source X' (NOT 'Library Source').\n\n")
            for i, result in enumerate(web_results, 1):
                title = result.get('title', 'Unknown')
                url = result.get('url', '')
                context_parts.append(f"[Web Source {i}: {title}]")
                if url:
                    context_parts.append(f"(URL: {url})")
                context_parts.append(result.get('snippet', ''))
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _get_feedback_context(self, question: str) -> List[Dict]:
        """
        BONUS: Get relevant feedback from self-learning storage.
        
        Args:
            question: Current question
            
        Returns:
            List of relevant feedback items
        """
        feedback_file = self.feedback_storage / "feedback_log.json"
        
        if not feedback_file.exists():
            return []
        
        try:
            with open(feedback_file, 'r') as f:
                all_feedback = json.load(f)
            
            # Find feedback for similar questions
            relevant = []
            question_lower = question.lower()
            
            for item in all_feedback:
                if any(keyword in question_lower for keyword in item.get('keywords', [])):
                    relevant.append(item)
            
            return relevant[:3]  # Top 3 most relevant
        
        except Exception as e:
            logger.error(f"Error loading feedback: {e}")
            return []
    
    def _generate_adaptive_answer(self, question: str, context: str,
                                  conversation_history: Optional[List[Dict]],
                                  feedback_context: List[Dict],
                                  has_local: bool, used_web: bool) -> str:
        """
        Generate adaptive answer with multi-step reasoning.
        
        Args:
            question: User's question
            context: Retrieved context
            conversation_history: Previous conversation
            feedback_context: Relevant feedback
            has_local: Has local results
            used_web: Used web search
            
        Returns:
            Generated answer
        """
        # Build conversation context
        conv_context = ""
        if conversation_history:
            conv_context = "\nRecent conversation:\n"
            for msg in conversation_history[-3:]:  # Last 3 messages
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                conv_context += f"{role.title()}: {content[:200]}\n"
        
        # Build feedback context (BONUS)
        feedback_text = ""
        if feedback_context:
            feedback_text = "\nLearned improvements from feedback:\n"
            for fb in feedback_context:
                feedback_text += f"- {fb.get('improvement', '')}\n"
        
        system_prompt = f"""You are an expert in Environmental Sustainability and Awareness.

Your approach:
1. Provide clear, adaptive explanations based on context
2. Use multi-step reasoning to break down complex concepts
3. Promote environmental awareness and actionable insights
4. If using web information, mention "according to recent information"
5. Be educational and engaging

{feedback_text}"""
        
        user_prompt = f"""{conv_context}

=== Retrieved Information from Knowledge Library ===
The following information has been retrieved from the knowledge base and web sources. Use relevant parts of this information to answer the question. You can use it for guidance, but prioritize accuracy and relevance to the question.

{context}

=== Question ===
{question}

=== CRITICAL INSTRUCTIONS FOR SOURCE REFERENCING ===
When you reference sources in your answer, you MUST use the EXACT labels from the sections above:

1. Sources in the "LIBRARY SOURCES" section:
   - These are labeled as "[Library Source 1: ...]", "[Library Source 2: ...]", etc.
   - Reference them as "Library Source 1", "Library Source 2", etc.

2. Sources in the "WEB SOURCES" section:
   - These are labeled as "[Web Source 1: ...]", "[Web Source 2: ...]", etc.
   - Reference them as "Web Source 1", "Web Source 2", etc.
   - NEVER call these "Library Source" - they are web search results, not library documents

3. If you see both sections above:
   - Count how many Library Sources are listed (e.g., if only "Library Source 1" exists, there is only 1 library source)
   - Count how many Web Sources are listed (e.g., if "Web Source 1" through "Web Source 4" exist, there are 4 web sources)
   - Do NOT continue the library numbering into the web sources section

=== Answer Instructions ===
Based on the retrieved information above, provide a comprehensive, well-reasoned answer to the question. 
- Use relevant parts from the retrieved information to support your answer
- Always use the correct source labels (Library Source X for library, Web Source X for web)
- If the retrieved information is relevant, incorporate it naturally into your response
- If the retrieved information is not directly relevant, you can still use your knowledge, but mention if you're going beyond the provided context
- Be clear, educational, and accurate"""
        
        answer = self.llm.generate_with_context(system_prompt, user_prompt, max_tokens=1500)
        
        if used_web and not has_local:
            answer += "\n\n*Note: This includes recent web information as local knowledge was limited.*"
        
        return answer
    
    def record_feedback(self, question: str, answer: str, 
                       feedback_type: str, feedback_text: str,
                       rating: Optional[int] = None):
        """
        BONUS: Record user feedback for self-learning.
        
        Args:
            question: Original question
            answer: Generated answer
            feedback_type: Type of feedback (positive/negative/suggestion)
            feedback_text: Feedback content
            rating: Optional rating (1-5)
        """
        feedback_file = self.feedback_storage / "feedback_log.json"
        
        # Load existing feedback
        if feedback_file.exists():
            with open(feedback_file, 'r') as f:
                all_feedback = json.load(f)
        else:
            all_feedback = []
        
        # Extract keywords from question
        keywords = [word.lower() for word in question.split() if len(word) > 4]
        
        # Create feedback entry
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer[:500],  # Store preview
            "feedback_type": feedback_type,
            "feedback_text": feedback_text,
            "rating": rating,
            "keywords": keywords[:5],  # Top 5 keywords
            "improvement": self._extract_improvement(feedback_text)
        }
        
        all_feedback.append(feedback_entry)
        
        # Save feedback
        with open(feedback_file, 'w') as f:
            json.dump(all_feedback, f, indent=2)
        
        logger.info(f"✓ Feedback recorded: {feedback_type}")
        
        # Analyze and generate improvements
        if len(all_feedback) % 10 == 0:  # Every 10 feedback items
            self._analyze_feedback_patterns(all_feedback)
    
    def _extract_improvement(self, feedback_text: str) -> str:
        """Extract actionable improvement from feedback."""
        # Simple extraction - in production, use LLM
        if "more detail" in feedback_text.lower():
            return "Provide more detailed explanations"
        elif "simpler" in feedback_text.lower():
            return "Use simpler language"
        elif "example" in feedback_text.lower():
            return "Include more examples"
        else:
            return feedback_text[:100]
    
    def _analyze_feedback_patterns(self, feedback_list: List[Dict]):
        """
        BONUS: Analyze feedback patterns for learning.
        
        Args:
            feedback_list: All feedback items
        """
        logger.info("Analyzing feedback patterns...")
        
        analysis_file = self.feedback_storage / "feedback_analysis.json"
        
        # Count feedback types
        patterns = {
            "total": len(feedback_list),
            "positive": sum(1 for f in feedback_list if f.get('feedback_type') == 'positive'),
            "negative": sum(1 for f in feedback_list if f.get('feedback_type') == 'negative'),
            "avg_rating": 0,
            "common_improvements": []
        }
        
        # Calculate average rating
        ratings = [f.get('rating', 0) for f in feedback_list if f.get('rating')]
        if ratings:
            patterns['avg_rating'] = sum(ratings) / len(ratings)
        
        # Find common improvements
        improvements = {}
        for f in feedback_list:
            imp = f.get('improvement', '')
            if imp:
                improvements[imp] = improvements.get(imp, 0) + 1
        
        patterns['common_improvements'] = sorted(
            improvements.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        # Save analysis
        with open(analysis_file, 'w') as f:
            json.dump(patterns, f, indent=2)
        
        logger.info(f"✓ Feedback analysis saved: {patterns['total']} items analyzed")
        if "more detail" in feedback_text.lower():
            return "Provide more detailed explanations"
        elif "simpler" in feedback_text.lower():
            return "Use simpler language"
        elif "example" in feedback_text.lower():
            return "Include more examples"
        else:
            return feedback_text[:100]
    
    def _analyze_feedback_patterns(self, feedback_list: List[Dict]):
        """
        BONUS: Analyze feedback patterns for learning.
        
        Args:
            feedback_list: All feedback items
        """
        logger.info("Analyzing feedback patterns...")
        
        analysis_file = self.feedback_storage / "feedback_analysis.json"
        
        # Count feedback types
        patterns = {
            "total": len(feedback_list),
            "positive": sum(1 for f in feedback_list if f.get('feedback_type') == 'positive'),
            "negative": sum(1 for f in feedback_list if f.get('feedback_type') == 'negative'),
            "avg_rating": 0,
            "common_improvements": []
        }
        
        # Calculate average rating
        ratings = [f.get('rating', 0) for f in feedback_list if f.get('rating')]
        if ratings:
            patterns['avg_rating'] = sum(ratings) / len(ratings)
        
        # Find common improvements
        improvements = {}
        for f in feedback_list:
            imp = f.get('improvement', '')
            if imp:
                improvements[imp] = improvements.get(imp, 0) + 1
        
        patterns['common_improvements'] = sorted(
            improvements.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        # Save analysis
        with open(analysis_file, 'w') as f:
            json.dump(patterns, f, indent=2)
        
        logger.info(f"✓ Feedback analysis saved: {patterns['total']} items analyzed")

