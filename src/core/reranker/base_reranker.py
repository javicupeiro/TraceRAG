from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

@dataclass
class RerankerResult:
    """Reranking result with score and metadata."""
    chunk_id: str
    score: float
    original_rank: int
    new_rank: int
    metadata: Dict[str, Any] = None

@dataclass
class RerankerResponse:
    """Complete reranking response."""
    results: List[RerankerResult]
    total_processed: int
    processing_time: float
    model_used: str
    method: str

class BaseReranker(ABC):
    """Base class for all rerankers."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: Optional[int] = None
    ) -> RerankerResponse:
        """
        Reorders documents based on query relevance.
        
        Args:
            query: User query
            documents: List of chunks with id, content, metadata
            top_k: Number of results to return
            
        Returns:
            RerankerResponse with reordered documents
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Checks if the reranker is working."""
        pass
