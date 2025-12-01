"""
BM25 Keyword-Based Retrieval
Implements BM25 algorithm for sparse retrieval
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Dict, Any, Optional
import logging
import numpy as np
from collections import Counter
import math
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    BM25 (Best Match 25) retrieval implementation.
    
    BM25 is a probabilistic ranking function that considers:
    - Term frequency (TF)
    - Inverse document frequency (IDF)
    - Document length normalization
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever.
        
        Args:
            k1: Term frequency saturation parameter (typically 1.2-2.0)
            b: Length normalization parameter (typically 0.75)
        """
        self.k1 = k1
        self.b = b
        
        # Index structures
        self.documents: List[Dict[str, Any]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.document_frequencies: Dict[str, int] = {}
        self.idf_scores: Dict[str, float] = {}
        self.inverted_index: Dict[str, List[int]] = {}
        
        self.is_indexed = False
        
        logger.info(f" BM25 Retriever initialized (k1={k1}, b={b})")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Simple tokenization - can be enhanced with stemming/lemmatization
        text = text.lower()
        # Remove punctuation and split
        tokens = []
        current_token = []
        
        for char in text:
            if char.isalnum():
                current_token.append(char)
            else:
                if current_token:
                    tokens.append(''.join(current_token))
                    current_token = []
        
        if current_token:
            tokens.append(''.join(current_token))
        
        return tokens
    
    def index_documents(self, documents: List[Dict[str, Any]], 
                       text_field: str = 'text'):
        """
        Index documents for BM25 retrieval.
        
        Args:
            documents: List of documents with text
            text_field: Field name containing text
        """
        logger.info(f"Indexing {len(documents)} documents for BM25...")
        
        self.documents = documents
        self.doc_lengths = []
        self.document_frequencies = Counter()
        self.inverted_index = {}
        
        # First pass: build inverted index and document frequencies
        for doc_id, doc in enumerate(documents):
            text = doc.get(text_field, '')
            tokens = self.tokenize(text)
            
            self.doc_lengths.append(len(tokens))
            
            # Track unique terms in this document
            unique_terms = set(tokens)
            for term in unique_terms:
                self.document_frequencies[term] += 1
                
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append(doc_id)
        
        # Calculate average document length
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # Calculate IDF scores
        num_docs = len(documents)
        for term, df in self.document_frequencies.items():
            # IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)
            self.idf_scores[term] = math.log((num_docs - df + 0.5) / (df + 0.5) + 1)
        
        self.is_indexed = True
        
        logger.info(f" Indexed {len(documents)} documents")
        logger.info(f"  - Vocabulary size: {len(self.document_frequencies)}")
        logger.info(f"  - Avg document length: {self.avg_doc_length:.1f} tokens")
    
    def calculate_bm25_score(self, query_tokens: List[str], doc_id: int) -> float:
        """
        Calculate BM25 score for a document given query tokens.
        
        Args:
            query_tokens: Tokenized query
            doc_id: Document ID
            
        Returns:
            BM25 score
        """
        if doc_id >= len(self.documents):
            return 0.0
        
        score = 0.0
        doc_length = self.doc_lengths[doc_id]
        
        # Get document text and tokenize
        doc_text = self.documents[doc_id].get('text', '')
        doc_tokens = self.tokenize(doc_text)
        doc_term_freqs = Counter(doc_tokens)
        
        # Calculate score for each query term
        for term in query_tokens:
            if term not in self.idf_scores:
                continue
            
            # Term frequency in document
            tf = doc_term_freqs.get(term, 0)
            
            if tf == 0:
                continue
            
            # IDF score
            idf = self.idf_scores[term]
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents using BM25.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with scores
        """
        if not self.is_indexed:
            logger.error("Documents not indexed. Call index_documents() first.")
            return []
        
        # Tokenize query
        query_tokens = self.tokenize(query)
        
        if not query_tokens:
            logger.warning("Empty query after tokenization")
            return []
        
        # Get candidate documents (union of all documents containing any query term)
        candidate_docs = set()
        for term in query_tokens:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term])
        
        # Calculate BM25 scores for candidates
        scored_docs = []
        for doc_id in candidate_docs:
            score = self.calculate_bm25_score(query_tokens, doc_id)
            
            if score > 0:
                result = {
                    **self.documents[doc_id],
                    'bm25_score': score,
                    'doc_id': doc_id
                }
                scored_docs.append(result)
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x['bm25_score'], reverse=True)
        
        logger.info(f"BM25 retrieved {len(scored_docs)} documents")
        
        return scored_docs[:top_k]
    
    def save_index(self, filepath: str):
        """Save BM25 index to disk."""
        index_data = {
            'k1': self.k1,
            'b': self.b,
            'documents': self.documents,
            'doc_lengths': self.doc_lengths,
            'avg_doc_length': self.avg_doc_length,
            'document_frequencies': dict(self.document_frequencies),
            'idf_scores': self.idf_scores,
            'inverted_index': self.inverted_index,
            'is_indexed': self.is_indexed
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
        
        logger.info(f" BM25 index saved to {filepath}")
    
    def load_index(self, filepath: str):
        """Load BM25 index from disk."""
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)
        
        self.k1 = index_data['k1']
        self.b = index_data['b']
        self.documents = index_data['documents']
        self.doc_lengths = index_data['doc_lengths']
        self.avg_doc_length = index_data['avg_doc_length']
        self.document_frequencies = Counter(index_data['document_frequencies'])
        self.idf_scores = index_data['idf_scores']
        self.inverted_index = index_data['inverted_index']
        self.is_indexed = index_data['is_indexed']
        
        logger.info(f" BM25 index loaded from {filepath}")
        logger.info(f"  - {len(self.documents)} documents indexed")
