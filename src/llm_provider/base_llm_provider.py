import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class LLMChunk:
    """
    Represents a single chunk that can be processed by an LLM provider.
    """
    content: str  # Text content, markdown table, or base64 image
    type: str     # 'text', 'table', or 'image'
    metadata: Dict[str, Any] = None  # Additional metadata like captions

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class LLMResponse:
    """
    Standardized response from LLM providers.
    """
    content: str  # Generated text response
    model_used: str  # Which model was used
    tokens_used: Optional[int] = None  # Number of tokens consumed
    response_time: Optional[float] = None  # Response time in seconds
    metadata: Dict[str, Any] = None  # Additional provider-specific data

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    
    This class defines the essential interface that any concrete LLM provider
    (e.g., GroqProvider, GeminiProvider) must implement.
    """

    def __init__(self, api_key: str, model_name: str, rate_limit_delay: float = 1.0):
        """
        Initialize the LLM provider.

        Args:
            api_key (str): API key for the LLM service
            model_name (str): Name of the model to use
            rate_limit_delay (float): Delay between requests in seconds for rate limiting
        """
        self.api_key = api_key
        self.model_name = model_name
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0.0
        
        logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")

    def _rate_limit(self):
        """
        Implement rate limiting by adding delay between requests.
        """
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()

    @abstractmethod
    def generate_summary(self, chunk: LLMChunk, prompt: str) -> LLMResponse:
        """
        Generate a summary for a single chunk.

        Args:
            chunk (LLMChunk): The chunk to summarize
            prompt (str): The prompt template for summarization

        Returns:
            LLMResponse: The generated summary response
        """
        pass

    @abstractmethod
    def answer_query(self, chunks: List[LLMChunk], prompt: str) -> LLMResponse:
        """
        Answer a user query based on multiple chunks.

        Args:
            chunks (List[LLMChunk]): List of relevant chunks (can be mixed types)
            prompt (str): The query prompt

        Returns:
            LLMResponse: The generated answer response
        """
        pass

    @abstractmethod
    def _prepare_messages(self, chunks: List[LLMChunk], prompt: str) -> List[Dict[str, Any]]:
        """
        Prepare messages format for the specific provider.

        Args:
            chunks (List[LLMChunk]): List of chunks to include
            prompt (str): The prompt text

        Returns:
            List[Dict[str, Any]]: Messages formatted for the provider's API
        """
        pass

    def _validate_chunks(self, chunks: List[LLMChunk]) -> bool:
        """
        Validate that chunks are properly formatted.

        Args:
            chunks (List[LLMChunk]): Chunks to validate

        Returns:
            bool: True if all chunks are valid
        """
        for chunk in chunks:
            if not isinstance(chunk, LLMChunk):
                logger.error(f"Invalid chunk type: {type(chunk)}")
                return False
            
            if chunk.type not in ['text', 'table', 'image']:
                logger.error(f"Invalid chunk type: {chunk.type}")
                return False
            
            if not chunk.content or not isinstance(chunk.content, str):
                logger.error(f"Invalid chunk content for type {chunk.type}")
                return False
        
        return True

    def _get_supported_modalities(self) -> List[str]:
        """
        Get list of supported modalities for this provider.
        Override in subclasses if different from default.

        Returns:
            List[str]: List of supported chunk types
        """
        return ['text', 'table', 'image']

    def health_check(self) -> bool:
        """
        Perform a health check to verify the provider is working.

        Returns:
            bool: True if provider is healthy
        """
        try:
            test_chunk = LLMChunk(content="Test", type="text")
            test_prompt = "Say 'OK' if you can read this."
            
            response = self.generate_summary(test_chunk, test_prompt)
            return bool(response.content)
        except Exception as e:
            logger.error(f"Health check failed for {self.__class__.__name__}: {e}")
            return False