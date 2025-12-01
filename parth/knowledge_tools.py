"""
Knowledge Retrieval Tools
Integrate RAG, hybrid retrieval, and search
"""

import sys
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, Any, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeTools:
    """
    Knowledge retrieval with:
    - RAG integration
    - Hybrid retrieval (Part G)
    - Web search fallback
    - Query enhancement
    - Result aggregation
    """
    
    def __init__(self, rag_system=None, hybrid_retriever=None, llm_client=None):
        """
        Initialize knowledge tools.
        
        Args:
            rag_system: RAG system from Part D
            hybrid_retriever: Hybrid retriever from Part G
            llm_client: LLM client
        """
        self.rag = rag_system
        self.hybrid = hybrid_retriever
        self.llm = llm_client
        
        # Query processor for enhancement
        if llm_client:
            try:
                from partg.query_processor import QueryProcessor
                self.query_processor = QueryProcessor(llm_client)
            except ImportError:
                self.query_processor = None
        else:
            self.query_processor = None
        
        self.retrieval_log: List[Dict] = []
        
        logger.info("✓ Knowledge Tools initialized")
    
    def retrieve_knowledge(self,
                          query: str,
                          method: str = "hybrid",
                          top_k: int = 5,
                          expand_query: bool = True,
                          similarity_cutoff: Optional[float] = None) -> Dict[str, Any]:
        """
        Retrieve knowledge using specified method.
        
        Args:
            query: Query text
            method: Retrieval method (hybrid, rag, vector, sparse)
            top_k: Number of results
            expand_query: Whether to expand query
            
        Returns:
            Retrieval results
        """
        logger.info(f"Retrieving knowledge: '{query}' (method={method})")
        
        # Query expansion
        queries = [query]
        if expand_query and self.query_processor:
            try:
                queries = self.query_processor.expand_query(query, method="synonyms")
                logger.info(f"  Expanded to {len(queries)} query variants")
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}")
        
        # Retrieve based on method
        if method == "hybrid" and self.hybrid:
            results = self._hybrid_retrieve(queries[0], top_k)
            # If hybrid returns no results, fallback to RAG/web search
            if results.get('success') and results.get('num_results', 0) == 0:
                logger.warning("Hybrid retrieval returned 0 results, falling back to RAG/web search")
                if self.rag:
                    results = self._rag_retrieve(queries[0], top_k)
                else:
                    results['error'] = 'No results found and RAG fallback not available'
        elif method == "rag" and self.rag:
            results = self._rag_retrieve(queries[0], top_k)
        elif method == "multi_query":
            # For multi-query, expand the query if processor available
            if expand_query and self.query_processor:
                try:
                    expanded = self.query_processor.expand_query(queries[0], method="synonyms")
                    if len(expanded) > 1:
                        queries = expanded
                except Exception as e:
                    logger.warning(f"Query expansion failed: {e}")
            
            # Use multiple query variants
            if len(queries) > 1:
                results = self._multi_query_retrieve(queries, top_k)
            else:
                # Fallback to hybrid if expansion didn't work, but preserve method name
                if self.hybrid:
                    hybrid_results = self._hybrid_retrieve(queries[0], top_k)
                    # If hybrid returns no results, fallback to RAG
                    if hybrid_results.get('success') and hybrid_results.get('num_results', 0) == 0 and self.rag:
                        logger.warning("Hybrid retrieval returned 0 results, falling back to RAG/web search")
                        results = self._rag_retrieve(queries[0], top_k)
                        results['method'] = 'multi_query'  # Preserve method name
                    else:
                        hybrid_results['method'] = 'multi_query'  # Preserve method name
                        results = hybrid_results
                elif self.rag:
                    results = self._rag_retrieve(queries[0], top_k)
                    results['method'] = 'multi_query'  # Preserve method name
                else:
                    results = {
                        'success': False,
                        'error': 'Multi-query requires query expansion or fallback method'
                    }
        else:
            # Try fallback to RAG if available
            if self.rag:
                logger.warning(f"Method {method} not available, falling back to RAG")
                results = self._rag_retrieve(queries[0], top_k)
            else:
                results = {
                    'success': False,
                    'error': f'Method {method} not available and RAG fallback not available'
                }
        
        # Apply similarity cutoff if specified
        if similarity_cutoff is not None and results.get('success'):
            # CRITICAL: Ensure query is in results for web search fallback
            # Always use the original user query, not any modified version
            if 'query' not in results:
                results['query'] = query
            # Also store in metadata for easy access
            if 'metadata' not in results:
                results['metadata'] = {}
            results['metadata']['query'] = query  # Store original query
            results = self._apply_similarity_cutoff(results, similarity_cutoff, top_k, query)
        
        # Log retrieval
        self._log_retrieval(query, method, results.get('success', False))
        
        return results
    
    def _hybrid_retrieve(self, query: str, top_k: int) -> Dict[str, Any]:
        """Retrieve using hybrid method (Part G)."""
        try:
            results = self.hybrid.hybrid_retrieve(query, top_k=top_k, rerank=True)
            
            # Format and enhance results
            formatted_results = []
            for result in results['results']:
                # Determine source type
                source_file = result.get('source_file', '')
                if source_file and source_file != 'web' and not source_file.startswith('http'):
                    source_type = 'library'
                else:
                    source_type = 'web'
                
                # Clean and format text
                text = result.get('text', '')
                text = self._clean_text(text)
                
                formatted_results.append({
                    'text': text,
                    'score': result.get('hybrid_score', result.get('rerank_score', result.get('score', 0))),
                    'source_file': source_file or 'unknown',
                    'source_type': source_type,
                    'chunk_id': result.get('chunk_id', ''),
                    'metadata': result.get('metadata', {})
                })
            
            # NOTE: Web results are NOT added here automatically
            # They will be added by _apply_similarity_cutoff if needed (when cutoff is enabled)
            # This ensures library is always checked first
            
            # Sort by score
            all_results = formatted_results
            all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            all_results = all_results[:top_k]
            
            # Generate answer using LLM with retrieved chunks (proper RAG)
            answer = ""
            if self.llm and all_results:
                # Build context from retrieved chunks
                context_parts = []
                
                # Separate library and web results (web may be added later by _apply_similarity_cutoff)
                library_results = [r for r in all_results[:5] if r.get('source_type') == 'library']
                web_results_list = [r for r in all_results[:5] if r.get('source_type') == 'web']
                
                # Add summary at the top
                context_parts.append("=== SOURCE SUMMARY ===\n")
                if web_results_list:
                    context_parts.append(f"Total sources provided: {len(library_results)} library source(s) and {len(web_results_list)} web source(s).\n")
                    context_parts.append("Library sources are numbered 1 to {}. Web sources are numbered 1 to {}.\n".format(len(library_results), len(web_results_list)))
                    context_parts.append("Do NOT mix the numbering - library and web sources have separate numbering systems.\n\n")
                else:
                    context_parts.append(f"Total sources provided: {len(library_results)} library source(s).\n")
                    context_parts.append("Library sources are numbered 1 to {}.\n\n".format(len(library_results)))
                
                # Add library sources section
                if library_results:
                    context_parts.append("=== LIBRARY SOURCES (from local knowledge base) ===\n")
                    context_parts.append("These are documents from the local knowledge library. Reference them as 'Library Source X'.\n\n")
                    for i, result in enumerate(library_results, 1):
                        source = result.get('source_file', 'Unknown')
                        score = result.get('score', 0)
                        context_parts.append(f"[Library Source {i}: {source}] (Relevance Score: {score:.2f})")
                        context_parts.append(result.get('text', '')[:1000])
                        context_parts.append("")
                
                # Add web sources section (only if web results exist)
                if web_results_list:
                    context_parts.append("\n=== WEB SOURCES (from internet search) ===\n")
                    context_parts.append("These are results from web search. Reference them as 'Web Source X' (NOT 'Library Source').\n\n")
                    for i, result in enumerate(web_results_list, 1):
                        title = result.get('title', 'Unknown')
                        url = result.get('url', result.get('source_file', 'Unknown'))
                        context_parts.append(f"[Web Source {i}: {title}] (URL: {url})")
                        context_parts.append(result.get('text', '')[:1000])
                        context_parts.append("")
                
                context = "\n".join(context_parts)
                
                # Generate answer using LLM with retrieved chunks
                prompt = f"""You are an expert in Environmental Sustainability and Awareness.

{context}

=== Question ===
{query}

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
                
                try:
                    answer = self.llm.generate(prompt, max_tokens=1500, temperature=0.7)
                except Exception as e:
                    logger.warning(f"Failed to generate answer with LLM: {e}")
                    answer = ""
            
            return {
                'success': True,
                'method': 'hybrid',
                'query': query,
                'results': all_results,
                'num_results': len(all_results),
                'answer': answer,  # Include generated answer
                'metadata': {
                    'num_dense': results.get('num_dense', 0),
                    'num_sparse': results.get('num_sparse', 0),
                    'alpha': results.get('alpha', 0.6),
                    'library_results': len(formatted_results),
                    'web_results': 0,  # Web results not added here, will be added by _apply_similarity_cutoff if needed
                    'answer_generated': bool(answer)
                }
            }
        
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _rag_retrieve(self, query: str, top_k: int) -> Dict[str, Any]:
        """Retrieve using RAG system (Part D) - ALWAYS checks library first."""
        try:
            # CRITICAL: Always retrieve from library first, don't use answer_question() which has web search logic
            # Directly use the retrieval pipeline to get library results
            if not self.rag or not self.rag.retrieval:
                return {
                    'success': False,
                    'error': 'RAG system or retrieval pipeline not available'
                }
            
            logger.info(f"Retrieving from library first for query: '{query}'")
            
            # Step 1: ALWAYS retrieve from library first (bypass answer_question's web search logic)
            local_results = self.rag.retrieval.hierarchical_retrieve(
                query,
                top_k=top_k * 2,  # Retrieve more to have options after filtering
                expand_context=True,
                rerank=True
            )
            
            # Convert library results to query format
            results = []
            library_results_count = 0
            
            # Check if Milvus is available
            metadata = local_results.get('metadata', {})
            milvus_available = metadata.get('milvus_available', True)
            
            if milvus_available and local_results.get('results'):
                # Add ALL library results (we'll filter by cutoff later if needed)
                for source in local_results.get('results', []):
                    results.append({
                        'text': self._clean_text(source.get('text', '')),
                        'score': source.get('rerank_score', source.get('score', 0)),
                        'source_file': source.get('source_file', 'library'),
                        'source_type': 'library',
                        'chunk_id': source.get('chunk_id', ''),
                        'metadata': source.get('metadata', {})
                    })
                    library_results_count += 1
                
                logger.info(f"Retrieved {library_results_count} library results from vector database")
            else:
                logger.warning("Milvus not available or no library results")
            
            # Step 2: Generate answer using library results (if available)
            answer = ""
            if self.llm and results:
                # Build context from library results
                context_parts = []
                context_parts.append("=== LIBRARY SOURCES (from local knowledge base) ===\n")
                context_parts.append("These are documents from the local knowledge library.\n\n")
                
                for i, result in enumerate(results[:5], 1):
                    source = result.get('source_file', 'Unknown')
                    score = result.get('score', 0)
                    context_parts.append(f"[Library Source {i}: {source}] (Relevance Score: {score:.2f})")
                    context_parts.append(result.get('text', '')[:1000])
                    context_parts.append("")
                
                context = "\n".join(context_parts)
                
                # Generate answer using LLM with library chunks
                prompt = f"""You are an expert in Environmental Sustainability and Awareness.

{context}

=== Question ===
{query}

=== Answer Instructions ===
Based on the retrieved information from the library above, provide a comprehensive, well-reasoned answer to the question.
- Use relevant parts from the retrieved information to support your answer
- Reference sources as "Library Source 1", "Library Source 2", etc.
- If the retrieved information is relevant, incorporate it naturally into your response
- Be clear, educational, and accurate"""
                
                try:
                    answer = self.llm.generate(prompt, max_tokens=1500, temperature=0.7)
                except Exception as e:
                    logger.warning(f"Failed to generate answer with LLM: {e}")
                    answer = ""
            
            # Sort results by score
            results.sort(key=lambda x: x.get('score', 0), reverse=True)
            results = results[:top_k]  # Take top_k library results
            
            # NOTE: Web results are NOT added here - they will be added by _apply_similarity_cutoff if needed
            # This ensures library is always checked first, web is only used as filler
            
            return {
                'success': True,
                'method': 'rag',
                'query': query,
                'results': results,
                'num_results': len(results),
                'answer': answer,
                'metadata': {
                    'milvus_available': milvus_available,
                    'library_results_count': library_results_count,
                    'retrieval_confidence': results[0].get('score', 0) if results else 0,
                    'used_web_search': False  # Web search not used at this stage
                }
            }
            
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }
    
    def _multi_query_retrieve(self, queries: List[str], top_k: int) -> Dict[str, Any]:
        """Retrieve using multiple query variants."""
        logger.info(f"Multi-query retrieval with {len(queries)} variants")
        
        all_results = []
        
        for query in queries:
            if self.hybrid:
                try:
                    results = self.hybrid.hybrid_retrieve(query, top_k=top_k // len(queries) + 1)
                    all_results.extend(results['results'])
                except Exception as e:
                    logger.warning(f"Query '{query}' failed: {e}")
        
        # Format and enhance results
        formatted_results = []
        for result in all_results:
            # Determine source type
            source_file = result.get('source_file', '')
            if source_file and source_file != 'web' and not source_file.startswith('http'):
                source_type = 'library'
            else:
                source_type = 'web'
            
            # Clean and format text
            text = result.get('text', '')
            text = self._clean_text(text)
            
            formatted_results.append({
                'text': text,
                'score': result.get('hybrid_score', result.get('rerank_score', result.get('score', 0))),
                'source_file': source_file or 'unknown',
                'source_type': source_type,
                'chunk_id': result.get('chunk_id', ''),
                'metadata': result.get('metadata', {})
            })
        
        # NOTE: Web results are NOT added here automatically
        # They will be added by _apply_similarity_cutoff if needed (when cutoff is enabled)
        # This ensures library is always checked first
        
        # Deduplicate and rank
        unique_results = self._deduplicate_results(formatted_results)
        unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Take top_k library results (web will be added later if needed)
        all_results = unique_results[:top_k]
        
        # Generate answer using LLM with retrieved chunks (proper RAG)
        answer = ""
        if self.llm and all_results:
            # Build context from retrieved chunks
            context_parts = []
            
            # Separate library and web results
            library_results = [r for r in all_results[:5] if r.get('source_type') == 'library']
            web_results_list = [r for r in all_results[:5] if r.get('source_type') == 'web']
            
            # Add summary at the top
            context_parts.append("=== SOURCE SUMMARY ===\n")
            context_parts.append(f"Total sources provided: {len(library_results)} library source(s) and {len(web_results_list)} web source(s).\n")
            context_parts.append("Library sources are numbered 1 to {}. Web sources are numbered 1 to {}.\n".format(len(library_results), len(web_results_list)))
            context_parts.append("Do NOT mix the numbering - library and web sources have separate numbering systems.\n\n")
            
            # Add library sources section
            if library_results:
                context_parts.append("=== LIBRARY SOURCES (from local knowledge base) ===\n")
                context_parts.append("These are documents from the local knowledge library. Reference them as 'Library Source X'.\n\n")
                for i, result in enumerate(library_results, 1):
                    source = result.get('source_file', 'Unknown')
                    score = result.get('score', 0)
                    context_parts.append(f"[Library Source {i}: {source}] (Relevance Score: {score:.2f})")
                    context_parts.append(result.get('text', '')[:1000])
                    context_parts.append("")
            
            # Add web sources section
            if web_results_list:
                context_parts.append("\n=== WEB SOURCES (from internet search) ===\n")
                context_parts.append("These are results from web search. Reference them as 'Web Source X' (NOT 'Library Source').\n\n")
                for i, result in enumerate(web_results_list, 1):
                    title = result.get('title', 'Unknown')
                    url = result.get('url', result.get('source_file', 'Unknown'))
                    context_parts.append(f"[Web Source {i}: {title}] (URL: {url})")
                    context_parts.append(result.get('text', '')[:1000])
                    context_parts.append("")
            
            context = "\n".join(context_parts)
            
            # Use the first query (original) for answer generation
            original_query = queries[0] if queries else ""
            
            # Generate answer using LLM with retrieved chunks
            prompt = f"""You are an expert in Environmental Sustainability and Awareness.

{context}

=== Question ===
{original_query}

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
            
            try:
                answer = self.llm.generate(prompt, max_tokens=1500, temperature=0.7)
            except Exception as e:
                logger.warning(f"Failed to generate answer with LLM: {e}")
                answer = ""
        
        return {
            'success': True,
            'method': 'multi_query',
            'queries': queries,
            'results': all_results,
            'num_results': len(all_results),
            'answer': answer,  # Include generated answer
            'metadata': {
                'library_results': len(unique_results),
                'web_results': 0,  # Web results not added here, will be added by _apply_similarity_cutoff if needed
                'answer_generated': bool(answer)
            }
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and format text for display."""
        if not text:
            return ""
        
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Split into sentences (preserving punctuation)
        # Pattern: split on sentence-ending punctuation followed by space or end
        sentences = re.split(r'([.!?]+\s+)', text)
        
        cleaned_sentences = []
        current_sentence = ""
        
        for part in sentences:
            part = part.strip()
            if not part:
                continue
            
            # If it's just punctuation, attach to previous
            if re.match(r'^[.!?]+$', part):
                if current_sentence:
                    current_sentence += part
                elif cleaned_sentences:
                    cleaned_sentences[-1] += part
            # If it ends with punctuation, it's a complete sentence
            elif part and part[-1] in '.!?':
                if current_sentence:
                    current_sentence += ' ' + part
                else:
                    current_sentence = part
                # Add complete sentence
                if current_sentence and len(current_sentence.strip()) > 10:  # Minimum length
                    cleaned_sentences.append(current_sentence.strip())
                    current_sentence = ""
            # Otherwise, it's a sentence fragment
            else:
                if current_sentence:
                    current_sentence += ' ' + part
                else:
                    current_sentence = part
        
        # Add any remaining sentence
        if current_sentence and len(current_sentence.strip()) > 10:
            cleaned_sentences.append(current_sentence.strip())
        
        # Join sentences with proper spacing
        if cleaned_sentences:
            cleaned_text = ' '.join(cleaned_sentences)
        else:
            # Fallback: if no complete sentences, return original (cleaned)
            cleaned_text = text
        
        # Final cleanup
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Ensure it starts with capital letter if possible
        if cleaned_text and cleaned_text[0].islower():
            cleaned_text = cleaned_text[0].upper() + cleaned_text[1:]
        
        return cleaned_text
    
    def _apply_similarity_cutoff(self, results: Dict[str, Any], cutoff: float, top_k: int, query: str = '') -> Dict[str, Any]:
        """
        Apply similarity score cutoff to filter library results and fill with web results.
        
        Args:
            results: Retrieval results dictionary
            cutoff: Similarity score cutoff (0-1)
            top_k: Desired number of total results
            
        Returns:
            Filtered results with library results above cutoff + web results filling remaining slots
        """
        if not results.get('success') or not results.get('results'):
            return results
        
        all_results = results['results']
        
        # Separate library and web results
        library_results = [r for r in all_results if r.get('source_type') == 'library']
        web_results = [r for r in all_results if r.get('source_type') == 'web']
        
        # Filter library results by cutoff
        filtered_library = [r for r in library_results if r.get('score', 0) >= cutoff]
        
        # Calculate how many slots we need to fill
        num_library = len(filtered_library)
        num_needed = max(0, top_k - num_library)
        
        logger.info(f"Similarity cutoff {cutoff}: {num_library} library results above cutoff, need {num_needed} more to reach top_k={top_k}")
        
        # If we need web results to fill slots and RAG is available, fetch them
        # Always fetch if num_needed > 0, regardless of existing web_results
        if num_needed > 0 and self.rag:
            try:
                logger.info(f"Fetching web results to fill {num_needed} slots (currently have {len(web_results)} web results)")
                # CRITICAL: Use the original query parameter first (most reliable)
                # Priority: query parameter > results['query'] > metadata['query']
                search_query = query  # Use the parameter passed to this function (original user query)
                if not search_query:
                    search_query = results.get('query', '')
                if not search_query:
                    # Try to get query from metadata
                    search_query = results.get('metadata', {}).get('query', '')
                
                # Clean the search query - ensure it's the original user query, not modified
                if search_query:
                    # Remove any system-added prefixes/suffixes that might interfere
                    original_query = search_query.strip()
                    # Only remove question words if they're at the start, keep the core topic
                    cleaned_query = original_query
                    question_prefixes = ["what is", "what are", "tell me about", "explain", "describe", "define"]
                    for prefix in question_prefixes:
                        if cleaned_query.lower().startswith(prefix):
                            cleaned_query = cleaned_query[len(prefix):].strip()
                            break
                    
                    # If cleaned query is too short, use original
                    if len(cleaned_query) < 3:
                        cleaned_query = original_query
                    
                    search_query = cleaned_query
                    # Log what we're searching for
                    logger.info(f"Web search query: '{search_query}' (original: '{original_query}')")
                
                if not search_query:
                    logger.warning("No query available for web search, cannot fetch web results")
                elif search_query:
                    # Force web search by directly calling RAG's web search method
                    try:
                        # Check if RAG has web search capability
                        has_web_search_method = hasattr(self.rag, '_perform_web_search')
                        web_search_available = getattr(self.rag, 'web_search_available', False)
                        
                        logger.info(f"RAG web search check: has_method={has_web_search_method}, available={web_search_available}")
                        
                        # Try to get web results directly from RAG's web search
                        if has_web_search_method and web_search_available:
                            logger.info(f"Calling RAG web search directly for: '{search_query}' (requesting {num_needed * 2} results)")
                            # Request more results to have options (request 2x to account for deduplication)
                            web_search_results = self.rag._perform_web_search(search_query, num_results=max(num_needed * 2, 5))
                            
                            logger.info(f"Web search returned {len(web_search_results)} results")
                            
                            for source in web_search_results:
                                # Check if we already have this web result
                                existing_urls = {r.get('url', r.get('source_file', '')) for r in web_results}
                                source_url = source.get('url', '')
                                
                                if source_url and source_url not in existing_urls:
                                    web_results.append({
                                        'text': self._clean_text(source.get('snippet', '')),
                                        'score': 0.5,  # Default score for web results
                                        'source_file': source_url,
                                        'source_type': 'web',
                                        'title': source.get('title', ''),
                                        'url': source_url
                                    })
                                    logger.debug(f"Added web result: {source.get('title', 'No title')[:50]}")
                                    
                                    if len(web_results) >= num_needed * 2:  # Have enough options
                                        break
                            
                            logger.info(f"Total web results after search: {len(web_results)} (need {num_needed})")
                        else:
                            # Fallback: Use answer_question which has web search built-in
                            logger.info(f"Direct web search method not available, using answer_question fallback")
                            # Save original threshold
                            original_threshold = getattr(self.rag, 'confidence_threshold', 0.3)
                            try:
                                # Set very high threshold to force web search
                                self.rag.confidence_threshold = 1.0
                                logger.info(f"Forcing web search by setting high confidence threshold")
                                rag_response = self.rag.answer_question(search_query, top_k=num_needed * 2)
                                web_sources = rag_response.get('sources', {}).get('web', [])
                                
                                logger.info(f"answer_question returned {len(web_sources)} web sources")
                                
                                # Add new web results
                                for source in web_sources:
                                    # Check if we already have this web result
                                    existing_urls = {r.get('url', r.get('source_file', '')) for r in web_results}
                                    source_url = source.get('url', '')
                                    
                                    if source_url and source_url not in existing_urls:
                                        web_results.append({
                                            'text': self._clean_text(source.get('snippet', '')),
                                            'score': 0.5,  # Default score for web results
                                            'source_file': source_url,
                                            'source_type': 'web',
                                            'title': source.get('title', ''),
                                            'url': source_url
                                        })
                                        
                                        if len(web_results) >= num_needed * 2:  # Have enough options
                                            break
                            finally:
                                # Restore original threshold
                                self.rag.confidence_threshold = original_threshold
                    except Exception as e2:
                        logger.error(f"Web search failed: {e2}")
                        import traceback
                        logger.error(traceback.format_exc())
            except Exception as e:
                logger.error(f"Failed to fetch additional web results: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Take top web results to fill remaining slots
        web_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # CRITICAL: Ensure we always return exactly top_k results
        # Priority: library above cutoff > web results > library below cutoff (last resort)
        
        # If we don't have enough web results, try to fetch more
        if len(web_results) < num_needed and self.rag:
            logger.warning(f"Only have {len(web_results)} web results but need {num_needed}, attempting to fetch more...")
            try:
                # Use original query parameter first (most reliable)
                search_query = query or results.get('query', '') or results.get('metadata', {}).get('query', '')
                if search_query:
                    # Clean the query (same logic as above)
                    original_query = search_query.strip()
                    cleaned_query = original_query
                    question_prefixes = ["what is", "what are", "tell me about", "explain", "describe", "define"]
                    for prefix in question_prefixes:
                        if cleaned_query.lower().startswith(prefix):
                            cleaned_query = cleaned_query[len(prefix):].strip()
                            break
                    if len(cleaned_query) < 3:
                        cleaned_query = original_query
                    search_query = cleaned_query
                    logger.info(f"Fetching additional web results with query: '{search_query}' (original: '{original_query}')")
                    
                    has_web_search_method = hasattr(self.rag, '_perform_web_search')
                    web_search_available = getattr(self.rag, 'web_search_available', False)
                    
                    if has_web_search_method and web_search_available:
                        # Try to get more web results
                        additional_needed = num_needed - len(web_results)
                        logger.info(f"Fetching {additional_needed} additional web results")
                        additional_results = self.rag._perform_web_search(search_query, num_results=max(additional_needed * 2, 5))
                        
                        existing_urls = {r.get('url', r.get('source_file', '')) for r in web_results}
                        for source in additional_results:
                            source_url = source.get('url', '')
                            if source_url and source_url not in existing_urls:
                                web_results.append({
                                    'text': self._clean_text(source.get('snippet', '')),
                                    'score': 0.5,
                                    'source_file': source_url,
                                    'source_type': 'web',
                                    'title': source.get('title', ''),
                                    'url': source_url
                                })
                                if len(web_results) >= num_needed:
                                    break
                        
                        # Re-sort
                        web_results.sort(key=lambda x: x.get('score', 0), reverse=True)
                        logger.info(f"After additional fetch: {len(web_results)} web results available (need {num_needed})")
            except Exception as e:
                logger.warning(f"Failed to fetch additional web results: {e}")
        
        # Select web results to fill remaining slots
        selected_web = web_results[:num_needed]
        logger.info(f"Selected {len(selected_web)} web results to fill {num_needed} slots (from {len(web_results)} available)")
        
        # Combine: library results (above cutoff) + web results (to fill slots)
        final_results = filtered_library + selected_web
        
        # If we still don't have exactly top_k results, try to get more web results
        if len(final_results) < top_k:
            remaining_needed = top_k - len(final_results)
            logger.info(f"Still need {remaining_needed} more results to reach top_k={top_k}, attempting to fetch more web results...")
            
            if self.rag:
                has_web_search_method = hasattr(self.rag, '_perform_web_search')
                web_search_available = getattr(self.rag, 'web_search_available', False)
                
                if has_web_search_method and web_search_available:
                    try:
                        # Use original query parameter first (most reliable)
                        search_query = query or results.get('query', '') or results.get('metadata', {}).get('query', '')
                        if search_query:
                            # Clean the query (same logic as above)
                            original_query = search_query.strip()
                            cleaned_query = original_query
                            question_prefixes = ["what is", "what are", "tell me about", "explain", "describe", "define"]
                            for prefix in question_prefixes:
                                if cleaned_query.lower().startswith(prefix):
                                    cleaned_query = cleaned_query[len(prefix):].strip()
                                    break
                            if len(cleaned_query) < 3:
                                cleaned_query = original_query
                            search_query = cleaned_query
                            logger.info(f"Additional web search query: '{search_query}' (original: '{original_query}')")
                            additional_web = self.rag._perform_web_search(search_query, num_results=remaining_needed * 2)
                            
                            existing_urls = {r.get('url', r.get('source_file', '')) for r in web_results}
                            for source in additional_web:
                                source_url = source.get('url', '')
                                if source_url and source_url not in existing_urls:
                                    web_results.append({
                                        'text': self._clean_text(source.get('snippet', '')),
                                        'score': 0.5,
                                        'source_file': source_url,
                                        'source_type': 'web',
                                        'title': source.get('title', ''),
                                        'url': source_url
                                    })
                                    if len(web_results) >= num_needed + remaining_needed:
                                        break
                            
                            # Re-sort and select more web results
                            web_results.sort(key=lambda x: x.get('score', 0), reverse=True)
                            additional_selected = web_results[num_needed:num_needed + remaining_needed]
                            final_results.extend(additional_selected)
                            selected_web.extend(additional_selected)
                            logger.info(f"Fetched {len(additional_selected)} additional web results: now have {len(final_results)} total results")
                    except Exception as e:
                        logger.warning(f"Failed to fetch additional web results: {e}")
            
            # CRITICAL: When cutoff is enabled, NEVER include library results below cutoff
            # Return fewer results rather than violating the cutoff threshold
            if len(final_results) < top_k:
                remaining_needed = top_k - len(final_results)
                logger.info(f"Respecting similarity cutoff: returning {len(final_results)} results instead of {top_k} (cutoff={cutoff})")
                logger.info(f"  - Library results above cutoff: {len(filtered_library)}")
                logger.info(f"  - Web results added: {len(selected_web)}")
                logger.info(f"  - Remaining slots not filled: {remaining_needed} (would require library results below cutoff)")
        
        # Sort all by score (library results first since they have higher scores, then web)
        # But ensure library results are always shown first regardless of score
        final_results.sort(key=lambda x: (
            0 if x.get('source_type') == 'library' else 1,  # Library first
            -x.get('score', 0)  # Then by score descending
        ))
        
        # Ensure we return at most top_k results (may be fewer if cutoff is strict)
        final_results = final_results[:top_k]
        
        # Preserve answer if it exists (generated before cutoff)
        existing_answer = results.get('answer', '')
        
        # Update results - preserve answer if it exists
        results['results'] = final_results
        results['num_results'] = len(final_results)
        
        # Ensure answer is preserved (don't remove it when applying cutoff)
        # The answer was generated before cutoff, so keep it
        if existing_answer:
            results['answer'] = existing_answer
        
        # Update metadata
        if 'metadata' not in results:
            results['metadata'] = {}
        results['metadata']['similarity_cutoff_applied'] = cutoff
        results['metadata']['library_results_above_cutoff'] = num_library
        results['metadata']['web_results_added'] = len(selected_web)
        results['metadata']['total_results'] = len(final_results)
        results['metadata']['cutoff_respected'] = True  # Indicate that cutoff was strictly enforced
        # Preserve existing metadata fields
        if 'used_web_search' not in results['metadata']:
            results['metadata']['used_web_search'] = len(selected_web) > 0
        
        # Final verification: log if we have fewer than top_k results
        if len(final_results) < top_k:
            logger.info(f"Returned {len(final_results)} results (requested top_k={top_k}) - cutoff={cutoff} strictly enforced, no library results below cutoff included")
        else:
            logger.info(f"✓ Successfully returned {len(final_results)} results (requested top_k={top_k})")
        
        logger.info(f"Applied cutoff {cutoff}: {num_library} library results above cutoff, {len(selected_web)} web results added, total: {len(final_results)} (cutoff strictly enforced)")
        
        return results
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results."""
        seen = set()
        unique = []
        
        for result in results:
            # Use text hash as key
            key = hash(result.get('text', ''))
            if key not in seen:
                seen.add(key)
                unique.append(result)
        
        return unique
    
    def aggregate_knowledge(self,
                          queries: List[str],
                          method: str = "hybrid") -> Dict[str, Any]:
        """
        Aggregate knowledge from multiple queries.
        
        Args:
            queries: List of queries
            method: Retrieval method
            
        Returns:
            Aggregated results
        """
        logger.info(f"Aggregating knowledge from {len(queries)} queries")
        
        all_results = []
        
        for query in queries:
            result = self.retrieve_knowledge(query, method=method, top_k=3)
            if result['success']:
                all_results.extend(result.get('results', []))
        
        # Deduplicate and rank
        unique_results = self._deduplicate_results(all_results)
        
        return {
            'success': True,
            'queries': queries,
            'total_retrieved': len(all_results),
            'unique_results': len(unique_results),
            'results': unique_results[:10]  # Top 10
        }
    
    def search_and_summarize(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search and generate summary.
        
        Args:
            query: Query text
            max_results: Maximum results to retrieve
            
        Returns:
            Search results with summary
        """
        logger.info(f"Search and summarize: '{query}'")
        
        # Retrieve
        retrieval = self.retrieve_knowledge(query, method="hybrid", top_k=max_results)
        
        if not retrieval['success']:
            return retrieval
        
        # Generate summary if LLM available
        if self.llm and retrieval.get('results'):
            try:
                # Build context
                context = "\n\n".join([
                    f"[{i+1}] {r.get('text', '')[:300]}"
                    for i, r in enumerate(retrieval['results'][:3])
                ])
                
                prompt = f"""Summarize the key information from these sources about: {query}

Sources:
{context}

Summary:"""
                
                summary = self.llm.generate(prompt, max_tokens=300, temperature=0.7)
                
                return {
                    'success': True,
                    'query': query,
                    'summary': summary,
                    'sources': retrieval['results'],
                    'num_sources': len(retrieval['results'])
                }
            
            except Exception as e:
                logger.error(f"Summarization failed: {e}")
                return {
                    'success': True,
                    'query': query,
                    'sources': retrieval['results'],
                    'summary_error': str(e)
                }
        
        return retrieval
    
    def _log_retrieval(self, query: str, method: str, success: bool):
        """Log retrieval."""
        self.retrieval_log.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'method': method,
            'success': success
        })
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        if not self.retrieval_log:
            return {'message': 'No retrievals logged'}
        
        total = len(self.retrieval_log)
        successful = sum(1 for log in self.retrieval_log if log['success'])
        
        by_method = {}
        for log in self.retrieval_log:
            method = log['method']
            by_method[method] = by_method.get(method, 0) + 1
        
        return {
            'total_retrievals': total,
            'successful': successful,
            'failed': total - successful,
            'success_rate': successful / total if total > 0 else 0,
            'by_method': by_method
        }
    def aggregate_knowledge(self,
                          queries: List[str],
                          method: str = "hybrid") -> Dict[str, Any]:
        """
        Aggregate knowledge from multiple queries.
        
        Args:
            queries: List of queries
            method: Retrieval method
            
        Returns:
            Aggregated results
        """
        logger.info(f"Aggregating knowledge from {len(queries)} queries")
        
        all_results = []
        
        for query in queries:
            result = self.retrieve_knowledge(query, method=method, top_k=3)
            if result['success']:
                all_results.extend(result.get('results', []))
        
        # Deduplicate and rank
        unique_results = self._deduplicate_results(all_results)
        
        return {
            'success': True,
            'queries': queries,
            'total_retrieved': len(all_results),
            'unique_results': len(unique_results),
            'results': unique_results[:10]  # Top 10
        }
    
    def search_and_summarize(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search and generate summary.
        
        Args:
            query: Query text
            max_results: Maximum results to retrieve
            
        Returns:
            Search results with summary
        """
        logger.info(f"Search and summarize: '{query}'")
        
        # Retrieve
        retrieval = self.retrieve_knowledge(query, method="hybrid", top_k=max_results)
        
        if not retrieval['success']:
            return retrieval
        
        # Generate summary if LLM available
        if self.llm and retrieval.get('results'):
            try:
                # Build context
                context = "\n\n".join([
                    f"[{i+1}] {r.get('text', '')[:300]}"
                    for i, r in enumerate(retrieval['results'][:3])
                ])
                
                prompt = f"""Summarize the key information from these sources about: {query}

Sources:
{context}

Summary:"""
                
                summary = self.llm.generate(prompt, max_tokens=300, temperature=0.7)
                
                return {
                    'success': True,
                    'query': query,
                    'summary': summary,
                    'sources': retrieval['results'],
                    'num_sources': len(retrieval['results'])
                }
            
            except Exception as e:
                logger.error(f"Summarization failed: {e}")
                return {
                    'success': True,
                    'query': query,
                    'sources': retrieval['results'],
                    'summary_error': str(e)
                }
        
        return retrieval
    
    def _log_retrieval(self, query: str, method: str, success: bool):
        """Log retrieval."""
        self.retrieval_log.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'method': method,
            'success': success
        })
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        if not self.retrieval_log:
            return {'message': 'No retrievals logged'}
        
        total = len(self.retrieval_log)
        successful = sum(1 for log in self.retrieval_log if log['success'])
        
        by_method = {}
        for log in self.retrieval_log:
            method = log['method']
            by_method[method] = by_method.get(method, 0) + 1
        
        return {
            'total_retrievals': total,
            'successful': successful,
            'failed': total - successful,
            'success_rate': successful / total if total > 0 else 0,
            'by_method': by_method
        }

