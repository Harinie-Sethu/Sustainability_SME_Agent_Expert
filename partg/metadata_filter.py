"""
Advanced Metadata Filtering
Intelligent filtering based on document metadata
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Dict, Any, Optional, Callable
import logging
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetadataFilter:
    """
    Advanced metadata filtering with:
    - Multi-field filtering
    - Range queries
    - Regex matching
    - Custom filter functions
    - Filter composition
    """
    
    def __init__(self):
        """Initialize metadata filter."""
        self.filters: List[Callable] = []
        logger.info(" Metadata Filter initialized")
    
    def add_exact_match_filter(self, field: str, value: Any):
        """
        Add exact match filter.
        
        Args:
            field: Metadata field name
            value: Value to match
        """
        def filter_func(result: Dict) -> bool:
            return result.get(field) == value
        
        self.filters.append(filter_func)
        logger.info(f"Added exact match filter: {field}={value}")
    
    def add_in_filter(self, field: str, values: List[Any]):
        """
        Add 'in' filter (field value must be in list).
        
        Args:
            field: Metadata field name
            values: List of acceptable values
        """
        def filter_func(result: Dict) -> bool:
            return result.get(field) in values
        
        self.filters.append(filter_func)
        logger.info(f"Added in filter: {field} in {values}")
    
    def add_range_filter(self, field: str, min_value: Optional[float] = None,
                        max_value: Optional[float] = None):
        """
        Add range filter.
        
        Args:
            field: Metadata field name
            min_value: Minimum value (inclusive)
            max_value: Maximum value (inclusive)
        """
        def filter_func(result: Dict) -> bool:
            value = result.get(field)
            if value is None:
                return False
            
            if min_value is not None and value < min_value:
                return False
            if max_value is not None and value > max_value:
                return False
            
            return True
        
        self.filters.append(filter_func)
        logger.info(f"Added range filter: {min_value} <= {field} <= {max_value}")
    
    def add_regex_filter(self, field: str, pattern: str):
        """
        Add regex pattern filter.
        
        Args:
            field: Metadata field name
            pattern: Regex pattern
        """
        compiled_pattern = re.compile(pattern)
        
        def filter_func(result: Dict) -> bool:
            value = result.get(field)
            if value is None:
                return False
            return bool(compiled_pattern.search(str(value)))
        
        self.filters.append(filter_func)
        logger.info(f"Added regex filter: {field} matches '{pattern}'")
    
    def add_contains_filter(self, field: str, substring: str, case_sensitive: bool = False):
        """
        Add substring contains filter.
        
        Args:
            field: Metadata field name
            substring: Substring to search for
            case_sensitive: Whether to match case
        """
        def filter_func(result: Dict) -> bool:
            value = result.get(field)
            if value is None:
                return False
            
            value_str = str(value)
            if not case_sensitive:
                value_str = value_str.lower()
                substring_lower = substring.lower()
                return substring_lower in value_str
            
            return substring in value_str
        
        self.filters.append(filter_func)
        logger.info(f"Added contains filter: {field} contains '{substring}'")
    
    def add_date_filter(self, field: str, after: Optional[datetime] = None,
                       before: Optional[datetime] = None):
        """
        Add date range filter.
        
        Args:
            field: Metadata field name (should contain ISO format date)
            after: Date must be after this
            before: Date must be before this
        """
        def filter_func(result: Dict) -> bool:
            value = result.get(field)
            if value is None:
                return False
            
            try:
                # Parse date
                if isinstance(value, str):
                    date_value = datetime.fromisoformat(value)
                elif isinstance(value, datetime):
                    date_value = value
                else:
                    return False
                
                if after and date_value < after:
                    return False
                if before and date_value > before:
                    return False
                
                return True
            except:
                return False
        
        self.filters.append(filter_func)
        logger.info(f"Added date filter: {after} < {field} < {before}")
    
    def add_custom_filter(self, filter_func: Callable[[Dict], bool], description: str = ""):
        """
        Add custom filter function.
        
        Args:
            filter_func: Function that takes result dict and returns bool
            description: Description of the filter
        """
        self.filters.append(filter_func)
        logger.info(f"Added custom filter: {description}")
    
    def apply_filters(self, results: List[Dict]) -> List[Dict]:
        """
        Apply all filters to results.
        
        Args:
            results: Results to filter
            
        Returns:
            Filtered results
        """
        if not self.filters:
            return results
        
        logger.info(f"Applying {len(self.filters)} filters to {len(results)} results")
        
        filtered_results = []
        for result in results:
            # Result must pass all filters
            if all(filter_func(result) for filter_func in self.filters):
                filtered_results.append(result)
        
        logger.info(f"  Filtered to {len(filtered_results)} results")
        
        return filtered_results
    
    def clear_filters(self):
        """Clear all filters."""
        self.filters = []
        logger.info("All filters cleared")
    
    def get_filter_count(self) -> int:
        """Get number of active filters."""
        return len(self.filters)


class MetadataScorer:
    """
    Score results based on metadata attributes.
    """
    
    def __init__(self):
        """Initialize metadata scorer."""
        self.scoring_rules: Dict[str, Any] = {}
        logger.info(" Metadata Scorer initialized")
    
    def add_categorical_scoring(self, field: str, score_map: Dict[Any, float]):
        """
        Add categorical scoring rule.
        
        Args:
            field: Metadata field
            score_map: Map from values to score multipliers
        """
        self.scoring_rules[field] = {
            'type': 'categorical',
            'score_map': score_map
        }
        logger.info(f"Added categorical scoring for {field}: {score_map}")
    
    def add_recency_scoring(self, field: str, decay_days: float = 365):
        """
        Add recency-based scoring (newer is better).
        
        Args:
            field: Date field
            decay_days: Number of days for score to decay to ~0.5
        """
        self.scoring_rules[field] = {
            'type': 'recency',
            'decay_days': decay_days
        }
        logger.info(f"Added recency scoring for {field} (decay={decay_days} days)")
    
    def add_numeric_scoring(self, field: str, min_val: float, max_val: float,
                           reverse: bool = False):
        """
        Add numeric scoring (normalize to 0-1).
        
        Args:
            field: Numeric field
            min_val: Minimum value
            max_val: Maximum value
            reverse: If True, lower is better
        """
        self.scoring_rules[field] = {
            'type': 'numeric',
            'min': min_val,
            'max': max_val,
            'reverse': reverse
        }
        logger.info(f"Added numeric scoring for {field}: [{min_val}, {max_val}]")
    
    def score_results(self, results: List[Dict]) -> List[Dict]:
        """
        Apply metadata scoring to results.
        
        Args:
            results: Results to score
            
        Returns:
            Results with metadata scores
        """
        logger.info(f"Applying metadata scoring with {len(self.scoring_rules)} rules")
        
        for result in results:
            metadata_score = 1.0
            score_components = {}
            
            for field, rule in self.scoring_rules.items():
                field_score = self._calculate_field_score(result, field, rule)
                metadata_score *= field_score
                score_components[field] = field_score
            
            result['metadata_score'] = metadata_score
            result['metadata_score_components'] = score_components
        
        return results
    
    def _calculate_field_score(self, result: Dict, field: str, rule: Dict) -> float:
        """Calculate score for a single field."""
        value = result.get(field)
        
        if value is None:
            return 1.0  # Neutral score if field missing
        
        rule_type = rule['type']
        
        if rule_type == 'categorical':
            score_map = rule['score_map']
            return score_map.get(value, 1.0)
        
        elif rule_type == 'recency':
            try:
                if isinstance(value, str):
                    date_value = datetime.fromisoformat(value)
                else:
                    date_value = value
                
                # Calculate days since date
                days_old = (datetime.now() - date_value).days
                decay_days = rule['decay_days']
                
                # Exponential decay
                import math
                score = math.exp(-days_old / decay_days)
                return max(0.1, min(2.0, score))  # Clamp to [0.1, 2.0]
            except:
                return 1.0
        
        elif rule_type == 'numeric':
            try:
                numeric_value = float(value)
                min_val = rule['min']
                max_val = rule['max']
                
                # Normalize to [0, 1]
                if max_val > min_val:
                    normalized = (numeric_value - min_val) / (max_val - min_val)
                    normalized = max(0.0, min(1.0, normalized))
                else:
                    normalized = 0.5
                
                # Reverse if needed
                if rule['reverse']:
                    normalized = 1.0 - normalized
                
                # Map to [0.5, 1.5] for scoring
                return 0.5 + normalized
            except:
                return 1.0
        
        return 1.0
