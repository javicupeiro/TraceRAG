import logging
import os
from typing import List, Dict, Any, Optional, Union
from .base_reranker import BaseReranker, RerankerResponse
from .jina_provider import JinaRerankerProvider

logger = logging.getLogger(__name__)

class RerankerHandler:
    """
    Main handler for the reranking system.
    Coordinates different providers and fallbacks.
    """
    
    def __init__(
        self,
        primary_provider: str = "jina",
        fallback_enabled: bool = True,
        **config
    ):
        self.primary_provider_name = primary_provider
        self.fallback_enabled = fallback_enabled
        self.config = config
        
        # Initialize primary provider
        self.primary_provider = self._create_provider(primary_provider, config)
        
        # Fallback provider (keep original order)
        self.fallback_provider = None
        
        logger.info(f"RerankerHandler initialized with primary: {primary_provider}")
    
    def _create_provider(self, provider_name: str, config: Dict[str, Any]) -> BaseReranker:
        """
        Factory to create reranking providers.
        """
        if provider_name.lower() == "jina":
            api_key = config.get('api_key') or os.getenv('JINA_API_KEY')
            if not api_key:
                raise ValueError("JINA_API_KEY not found in config or environment")
            
            return JinaRerankerProvider(
                api_key=api_key,
                model_name=config.get('model_name', 'jina-reranker-v2-base-multilingual'),
                **config
            )
        
        else:
            raise ValueError(f"Unknown reranker provider: {provider_name}")
    
    def rerank_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> RerankerResponse:
        """
        Reorders chunks based on query relevance.
        
        Args:
            query: User query
            chunks: List of chunks from RAG retrieval
            top_k: Number of final results to return
            
        Returns:
            RerankerResponse with reordered chunks
        """
        if not chunks:
            logger.warning("No chunks provided for reranking")
            return RerankerResponse(
                results=[],
                total_processed=0,
                processing_time=0.0,
                model_used="none",
                method="empty_input"
            )
        
        # Try with primary provider
        try:
            response = self.primary_provider.rerank(query, chunks, top_k)
            
            # If reranking succeeded, return
            if response.method != "fallback":
                return response
                
        except Exception as e:
            logger.error(f"Primary reranker failed: {e}")
        
        # Fallback: keep original order
        if self.fallback_enabled:
            logger.info("Using fallback reranking (original order)")
            return self._fallback_rerank(chunks, top_k)
        else:
            raise Exception("Reranking failed and fallback is disabled")
    
    def _fallback_rerank(
        self, 
        chunks: List[Dict[str, Any]], 
        top_k: Optional[int] = None
    ) -> RerankerResponse:
        """
        Fallback that keeps the original order of chunks.
        """
        from .base_reranker import RerankerResult
        
        results = []
        for i, chunk in enumerate(chunks):
            result = RerankerResult(
                chunk_id=chunk.get('id', f'chunk_{i}'),
                score=1.0 - (i * 0.01),  # Artificially decreasing score
                original_rank=i,
                new_rank=i,
                metadata={'fallback': True, 'original_doc': chunk}
            )
            results.append(result)
        
        # Apply top_k if specified
        if top_k:
            results = results[:top_k]
        
        return RerankerResponse(
            results=results,
            total_processed=len(chunks),
            processing_time=0.0,
            model_used="fallback",
            method="original_order"
        )
    
    def health_check(self) -> Dict[str, bool]:
        """
        Verifies the status of all providers.
        """
        status = {}
        
        try:
            status['primary'] = self.primary_provider.health_check()
        except Exception as e:
            logger.error(f"Primary provider health check failed: {e}")
            status['primary'] = False
        
        status['fallback'] = self.fallback_enabled
        
        return status
    
    def get_reranked_chunks(
        self, 
        reranker_response: RerankerResponse
    ) -> List[Dict[str, Any]]:
        """
        Converts RerankerResponse back into a list of ordered chunks.
        """
        reranked_chunks = []
        
        for result in reranker_response.results:
            # Retrieve original chunk
            original_chunk = result.metadata.get('original_doc', {})
            
            # Add reranking information
            enhanced_chunk = original_chunk.copy()
            enhanced_chunk['rerank_score'] = result.score
            enhanced_chunk['original_rank'] = result.original_rank
            enhanced_chunk['new_rank'] = result.new_rank
            
            reranked_chunks.append(enhanced_chunk)
        
        return reranked_chunks
