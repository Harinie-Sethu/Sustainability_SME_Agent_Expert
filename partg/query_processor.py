"""
Query Processing and Expansion
Enhance queries for better retrieval
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Dict, Any, Optional
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    Query processing for improved retrieval:
    - Query expansion
    - Synonym addition
    - Query reformulation
    - Multi-query generation
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize query processor.
        
        Args:
            llm_client: Optional LLM for query expansion
        """
        self.llm = llm_client
        
        # Domain-specific synonyms for environmental topics
        self.synonym_map = {
            'climate change': ['global warming', 'climate crisis', 'climate emergency'],
            'renewable energy': ['clean energy', 'green energy', 'sustainable energy'],
            'pollution': ['contamination', 'environmental degradation'],
            'conservation': ['preservation', 'protection', 'sustainability'],
            'biodiversity': ['biological diversity', 'ecosystem diversity'],
            'deforestation': ['forest loss', 'forest clearance'],
            'greenhouse gas': ['GHG', 'carbon emissions', 'emissions'],
            'solar power': ['solar energy', 'photovoltaic', 'solar panels'],
            'wind power': ['wind energy', 'wind turbines'],
            'recycling': ['waste recycling', 'material recovery'],
            'composting': ['organic waste composting', 'compost'],
            'carbon footprint': ['carbon emissions', 'CO2 footprint']
        }
        
        logger.info(" Query Processor initialized")
    
    def expand_query(self, query: str, method: str = "synonyms") -> List[str]:
        """
        Expand query to improve retrieval coverage.
        
        Args:
            query: Original query
            method: Expansion method ("synonyms", "llm", "multi_query")
            
        Returns:
            List of query variants
        """
        logger.info(f"Expanding query: '{query}' (method={method})")
        
        if method == "synonyms":
            expanded = self._expand_with_synonyms(query)
        elif method == "llm" and self.llm:
            expanded = self._expand_with_llm(query)
        elif method == "multi_query" and self.llm:
            expanded = self._generate_multi_queries(query)
        else:
            expanded = [query]
        
        logger.info(f"  Expanded to {len(expanded)} variants")
        
        return expanded
    
    def _expand_with_synonyms(self, query: str) -> List[str]:
        """Expand query with domain synonyms."""
        query_lower = query.lower()
        expanded_queries = [query]  # Always include original
        
        # Find matching terms and add synonyms
        for term, synonyms in self.synonym_map.items():
            if term in query_lower:
                # Replace term with each synonym
                for synonym in synonyms:
                    expanded_query = re.sub(
                        re.escape(term),
                        synonym,
                        query_lower,
                        flags=re.IGNORECASE
                    )
                    if expanded_query not in [q.lower() for q in expanded_queries]:
                        expanded_queries.append(expanded_query)
        
        return expanded_queries
    
    def _expand_with_llm(self, query: str) -> List[str]:
        """Expand query using LLM."""
        prompt = f"""Generate 3 alternative phrasings of this query that maintain the same meaning:

Original Query: {query}

Alternative phrasings (one per line):
1."""
        
        try:
            response = self.llm.generate(prompt, temperature=0.7, max_tokens=200)
            
            # Parse alternatives
            alternatives = [query]  # Include original
            for line in response.split('\n'):
                line = line.strip()
                # Remove numbering
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                if line and len(line) > 10 and line not in alternatives:
                    alternatives.append(line)
            
            return alternatives[:4]  # Original + 3 alternatives
        
        except Exception as e:
            logger.error(f"LLM query expansion failed: {e}")
            return [query]
    
    def _generate_multi_queries(self, query: str) -> List[str]:
        """Generate multiple related queries."""
        prompt = f"""Given this question about environmental sustainability, generate 3 related questions that would help answer it comprehensively:

Original Question: {query}

Related questions:
1."""
        
        try:
            response = self.llm.generate(prompt, temperature=0.7, max_tokens=300)
            
            # Parse questions
            questions = [query]  # Include original
            for line in response.split('\n'):
                line = line.strip()
                # Remove numbering
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                if line and len(line) > 10 and '?' in line:
                    questions.append(line)
            
            return questions[:4]  # Original + 3 related
        
        except Exception as e:
            logger.error(f"Multi-query generation failed: {e}")
            return [query]
    
    def extract_key_terms(self, query: str) -> List[str]:
        """
        Extract key terms from query.
        
        Args:
            query: Query text
            
        Returns:
            List of key terms
        """
        # Remove common stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'what', 'how', 'why', 'when', 'where', 'which', 'who', 'does', 'do',
            'can', 'could', 'should', 'would', 'will'
        }
        
        # Tokenize and filter
        tokens = re.findall(r'\b\w+\b', query.lower())
        key_terms = [t for t in tokens if t not in stopwords and len(t) > 3]
        
        return key_terms
    
    def reformulate_query(self, query: str, context: Optional[str] = None) -> str:
        """
        Reformulate query for better retrieval.
        
        Args:
            query: Original query
            context: Optional context from conversation
            
        Returns:
            Reformulated query
        """
        # If question is very short, expand it
        if len(query.split()) <= 3:
            key_terms = self.extract_key_terms(query)
            if key_terms:
                reformulated = f"Information about {' '.join(key_terms)} in environmental sustainability"
                logger.info(f"Reformulated short query: '{query}' -> '{reformulated}'")
                return reformulated
        
        return query
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze query intent and type.
        
        Args:
            query: Query text
            
        Returns:
            Analysis results
        """
        query_lower = query.lower()
        
        analysis = {
            'query': query,
            'type': 'general',
            'requires_comparison': False,
            'requires_explanation': False,
            'requires_examples': False,
            'temporal_aspect': None,
            'key_terms': self.extract_key_terms(query)
        }
        
        # Detect query type
        if any(word in query_lower for word in ['what', 'define', 'explain']):
            analysis['type'] = 'definition'
            analysis['requires_explanation'] = True
        
        elif any(word in query_lower for word in ['how', 'process', 'steps']):
            analysis['type'] = 'procedural'
            analysis['requires_explanation'] = True
        
        elif any(word in query_lower for word in ['why', 'reason', 'cause']):
            analysis['type'] = 'causal'
            analysis['requires_explanation'] = True
        
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            analysis['type'] = 'comparison'
            analysis['requires_comparison'] = True
        
        elif any(word in query_lower for word in ['example', 'instance', 'case']):
            analysis['requires_examples'] = True
        
        # Detect temporal aspects
        if any(word in query_lower for word in ['future', 'will', 'predict']):
            analysis['temporal_aspect'] = 'future'
        elif any(word in query_lower for word in ['past', 'history', 'was']):
            analysis['temporal_aspect'] = 'past'
        elif any(word in query_lower for word in ['current', 'now', 'today', 'recent']):
            analysis['temporal_aspect'] = 'present'
        
        return analysis
