import logging
import time
import requests
from typing import List, Dict, Any, Optional
from .base_reranker import BaseReranker, RerankerResult, RerankerResponse

logger = logging.getLogger(__name__)

class JinaRerankerProvider(BaseReranker):
    """
    Implementation of reranker using Jina v2 API.
    """
    
    def __init__(
        self, 
        api_key: str,
        model_name: str = "jina-reranker-v2-base-multilingual",
        api_base: str = "https://api.jina.ai/v1/rerank",
        max_retries: int = 3,
        timeout: int = 30,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key
        self.api_base = api_base
        self.max_retries = max_retries
        self.timeout = timeout
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"JinaRerankerProvider initialized with model: {model_name}")
    
    def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: Optional[int] = None
    ) -> RerankerResponse:
        """
        Reorders documents using Jina Reranker v2 API.
        """
        start_time = time.time()
        
        if not documents:
            return RerankerResponse(
                results=[],
                total_processed=0,
                processing_time=0.0,
                model_used=self.model_name,
                method="jina_api"
            )
        
        # Prepare documents for the API
        doc_texts = []
        doc_metadata = []
        
        for i, doc in enumerate(documents):
            # Extract text content from document
            content = self._extract_text_content(doc)
            doc_texts.append(content)
            doc_metadata.append({
                'original_index': i,
                'chunk_id': doc.get('id', f'chunk_{i}'),
                'chunk_type': doc.get('chunk_type', 'unknown'),
                'original_doc': doc
            })
        
        # Call Jina API
        try:
            scores = self._call_jina_api(query, doc_texts)
            
            # Process results
            results = []
            for i, (score, metadata) in enumerate(zip(scores, doc_metadata)):
                result = RerankerResult(
                    chunk_id=metadata['chunk_id'],
                    score=score,
                    original_rank=metadata['original_index'],
                    new_rank=i,  # Will be updated after sorting
                    metadata=metadata
                )
                results.append(result)
            
            # Sort by score descending
            results.sort(key=lambda x: x.score, reverse=True)
            
            # Update new_rank after sorting
            for new_rank, result in enumerate(results):
                result.new_rank = new_rank
            
            # Apply top_k if specified
            if top_k:
                results = results[:top_k]
            
            processing_time = time.time() - start_time
            
            logger.info(
                f"Jina reranking complete: {len(documents)} docs -> {len(results)} results "
                f"(time: {processing_time:.2f}s)"
            )
            
            return RerankerResponse(
                results=results,
                total_processed=len(documents),
                processing_time=processing_time,
                model_used=self.model_name,
                method="jina_api"
            )
            
        except Exception as e:
            logger.error(f"Jina reranking failed: {e}")
            # Fallback: maintain original order
            return self._fallback_response(documents, start_time)
    
    def _extract_text_content(self, document: Dict[str, Any]) -> str:
        """
        Extracts relevant text content from document for reranking.
        """
        content = document.get('content', '')
        chunk_type = document.get('chunk_type', 'text')
        
        if chunk_type == 'text':
            return content
        
        elif chunk_type in ['image', 'table']:
            # For images and tables, use the summary if available
            summary = document.get('summary', '')
            if summary:
                return summary
            
            # Fallback: use caption if exists
            metadata = document.get('chunk_metadata', {})
            caption = metadata.get('caption', '')
            if caption:
                return f"Content description: {caption}"
            
            return f"{chunk_type.title()} content (visual element)"
        
        elif chunk_type == 'code':
            # For code, include the language if available
            metadata = document.get('chunk_metadata', {})
            language = metadata.get('language', '')
            if language:
                return f"{language} code: {content}"
            return f"Code: {content}"
        
        return content
    
    def _call_jina_api(self, query: str, documents: List[str]) -> List[float]:
        """
        Calls Jina API to get reranking scores.
        """
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents
            # Note: top_k parameter is not supported by Jina API
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_base,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 401:
                    raise Exception("Unauthorized: Invalid Jina API key. Please check your JINA_API_KEY environment variable.")
                
                response.raise_for_status()
                
                data = response.json()
                
                # Extract scores correctly from the response
                results = data.get('results', [])
                
                # Create a mapping from original index to score
                score_map = {}
                for result in results:
                    original_index = result.get('index')
                    score = result.get('relevance_score', 0.0)
                    score_map[original_index] = score
                
                # Return scores in original document order
                scores = []
                for i in range(len(documents)):
                    scores.append(score_map.get(i, 0.0))
                
                return scores
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Jina API attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1 * (attempt + 1))  # Exponential backoff
        
        raise Exception("Jina API failed after all retries")
    
    def _fallback_response(
        self, 
        documents: List[Dict[str, Any]], 
        start_time: float
    ) -> RerankerResponse:
        """
        Fallback response when reranking fails.
        Maintains original order.
        """
        results = []
        for i, doc in enumerate(documents):
            result = RerankerResult(
                chunk_id=doc.get('id', f'chunk_{i}'),
                score=1.0 - (i * 0.01),  # Decreasing score
                original_rank=i,
                new_rank=i,
                metadata={'fallback': True, 'original_doc': doc}
            )
            results.append(result)
        
        return RerankerResponse(
            results=results,
            total_processed=len(documents),
            processing_time=time.time() - start_time,
            model_used="fallback",
            method="original_order"
        )
    
    def health_check(self) -> bool:
        """
        Verifies that Jina API is working.
        """
        try:
            test_query = "test query"
            test_docs = ["test document"]
            
            self._call_jina_api(test_query, test_docs)
            return True
            
        except Exception as e:
            logger.error(f"Jina health check failed: {e}")
            return False
        