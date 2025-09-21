import logging
import time
from typing import List, Dict, Any
import requests
import json

from .base_llm_provider import BaseLLMProvider, LLMChunk, LLMResponse

# Get a logger for the current module
logger = logging.getLogger(__name__)

class GroqProvider(BaseLLMProvider):
    """
    Groq LLM provider implementation.
    
    Supports text processing through Groq's API.
    Note: Groq currently supports primarily text-based models.
    """

    def __init__(self, api_key: str, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct", 
                 rate_limit_delay: float = 1.0, temperature: float = 0.1, max_output_tokens: int = 1024):
        """
        Initialize the Groq provider.

        Args:
            api_key (str): Groq API key
            model_name (str): Model name (default: meta-llama/llama-4-scout-17b-16e-instruct for vision)
            rate_limit_delay (float): Delay between requests for rate limiting
            temperature (float): Temperature for response generation (0.0-2.0)
            max_output_tokens (int): Maximum output tokens per response
        """
        super().__init__(api_key, model_name, rate_limit_delay)
        # Groq uses OpenAI-compatible API format
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Store generation parameters
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        # Check if model supports vision
        self.supports_vision = "llama-4-scout" in model_name or "vision" in model_name.lower()
        
        logger.info(f"GroqProvider initialized with model: {model_name} (Vision: {self.supports_vision}, "
                   f"Temp: {temperature}, Max tokens: {max_output_tokens})")

    def _get_supported_modalities(self) -> List[str]:
        """
        Groq supports text, tables and images (with vision models).

        Returns:
            List[str]: Supported chunk types
        """
        if self.supports_vision:
            return ['text', 'table', 'image']
        else:
            return ['text', 'table']  # Non-vision models

    def generate_summary(self, chunk: LLMChunk, prompt: str) -> LLMResponse:
        """
        Generate a summary for a single chunk using Groq API.

        Args:
            chunk (LLMChunk): The chunk to summarize
            prompt (str): The prompt template for summarization

        Returns:
            LLMResponse: The generated summary response
        """
        if not self._validate_chunks([chunk]):
            raise ValueError("Invalid chunk provided")
        
        # Additional validation for images with vision models
        if chunk.type == 'image' and self.supports_vision:
            if not self._validate_image_size(chunk.content):
                raise ValueError("Image size exceeds Groq limits (4MB for base64)")

        self._rate_limit()
        
        start_time = time.time()
        
        try:
            messages = self._prepare_messages([chunk], prompt)
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.3,
                "max_completion_tokens": 512,  # Use max_completion_tokens instead of max_tokens for vision models
                "top_p": 1,
                "stream": False
            }
            
            logger.debug(f"Sending summary request to Groq for chunk type: {chunk.type}")
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            content = response_data['choices'][0]['message']['content']
            tokens_used = response_data.get('usage', {}).get('total_tokens', 0)
            
            response_time = time.time() - start_time
            
            logger.info(f"Successfully generated summary using Groq (tokens: {tokens_used}, time: {response_time:.2f}s)")
            
            return LLMResponse(
                content=content,
                model_used=self.model_name,
                tokens_used=tokens_used,
                response_time=response_time,
                metadata={"provider": "groq"}
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error generating summary with Groq: {e}")
            raise

    def answer_query(self, chunks: List[LLMChunk], prompt: str) -> LLMResponse:
        """
        Answer a user query based on multiple chunks using Groq API.

        Args:
            chunks (List[LLMChunk]): List of relevant chunks
            prompt (str): The query prompt

        Returns:
            LLMResponse: The generated answer response
        """
        if not chunks:
            raise ValueError("No chunks provided")
        
        if not self._validate_chunks(chunks):
            raise ValueError("Invalid chunks provided")
        
        # Additional validation for images with vision models
        for chunk in chunks:
            if chunk.type == 'image' and self.supports_vision:
                if not self._validate_image_size(chunk.content):
                    raise ValueError(f"Image size exceeds Groq limits (4MB for base64)")

        self._rate_limit()
        
        start_time = time.time()
        
        try:
            messages = self._prepare_messages(chunks, prompt)
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_completion_tokens": self.max_output_tokens,
                "top_p": 1,
                "stream": False
            }
            
            logger.debug(f"Sending query request to Groq with {len(chunks)} chunks")
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            content = response_data['choices'][0]['message']['content']
            tokens_used = response_data.get('usage', {}).get('total_tokens', 0)
            
            response_time = time.time() - start_time
            
            logger.info(f"Successfully answered query using Groq (tokens: {tokens_used}, time: {response_time:.2f}s)")
            
            return LLMResponse(
                content=content,
                model_used=self.model_name,
                tokens_used=tokens_used,
                response_time=response_time,
                metadata={"provider": "groq", "chunks_processed": len(chunks)}
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error answering query with Groq: {e}")
            raise

    def _prepare_messages(self, chunks: List[LLMChunk], prompt: str) -> List[Dict[str, Any]]:
        """
        Prepare messages for Groq API format (OpenAI-compatible).

        Args:
            chunks (List[LLMChunk]): List of chunks to include
            prompt (str): The prompt text

        Returns:
            List[Dict[str, Any]]: Messages formatted for Groq API
        """
        
        if len(chunks) == 1:
            # Single chunk case (usually for summarization)
            chunk = chunks[0]
            
            if chunk.type == 'image' and self.supports_vision:
                # Vision model - send image directly
                data_url = self._prepare_image_data_url(chunk.content)
                
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant that can analyze images and provide accurate descriptions."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]
                    }
                ]
            else:
                # Text or table content
                if chunk.type == 'image' and not self.supports_vision:
                    # Fallback for non-vision models
                    description = chunk.metadata.get('caption', 'Image content')
                    content = f"{prompt}\n\nImage description: {description}"
                else:
                    content = f"{prompt}\n\nContent: {chunk.content}"
                
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. Provide accurate and concise responses based on the given context."
                    },
                    {
                        "role": "user", 
                        "content": content
                    }
                ]
        else:
            # Multiple chunks case (usually for query answering)
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Analyze the provided content and answer the user's question accurately."
                }
            ]
            
            # Build content array for mixed media
            content_parts = [{"type": "text", "text": prompt + "\n\nContext:"}]
            
            for i, chunk in enumerate(chunks):
                if chunk.type == 'text':
                    content_parts.append({
                        "type": "text", 
                        "text": f"\n\nTEXT CHUNK {i+1}:\n{chunk.content}"
                    })
                elif chunk.type == 'table':
                    content_parts.append({
                        "type": "text",
                        "text": f"\n\nTABLE {i+1}:\n{chunk.content}"
                    })
                elif chunk.type == 'image':
                    if self.supports_vision:
                        # Add image directly
                        data_url = self._prepare_image_data_url(chunk.content)
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": data_url}
                        })
                        # Add caption if available
                        caption = chunk.metadata.get('caption', '')
                        if caption:
                            content_parts.append({
                                "type": "text",
                                "text": f"\nImage {i+1} caption: {caption}"
                            })
                    else:
                        # Fallback for non-vision models
                        description = chunk.metadata.get('caption', 'Image content')
                        content_parts.append({
                            "type": "text",
                            "text": f"\n\nIMAGE {i+1} DESCRIPTION:\n{description}"
                        })
                        logger.warning(f"Image chunk {i+1} converted to text description (non-vision model)")
            
            messages.append({
                "role": "user",
                "content": content_parts
            })
        
        return messages

    def _prepare_image_data_url(self, base64_data: str) -> str:
        """
        Prepare image data URL for Groq API.
        
        Args:
            base64_data (str): Base64 encoded image data
            
        Returns:
            str: Data URL formatted for Groq API
        """
        # Validate and get MIME type
        mime_type = self._detect_image_mime_type(base64_data)
        
        # Check if already formatted as data URL
        if base64_data.startswith('data:'):
            return base64_data
        
        # Create data URL
        return f"data:{mime_type};base64,{base64_data}"

    def _detect_image_mime_type(self, base64_data: str) -> str:
        """
        Detect MIME type of base64 image data.
        
        Args:
            base64_data (str): Base64 encoded image data
            
        Returns:
            str: MIME type of the image
        """
        try:
            import base64
            # Remove data URL prefix if present
            if base64_data.startswith('data:'):
                base64_data = base64_data.split(',')[1]
            
            decoded = base64.b64decode(base64_data[:100])  # Check first bytes
            
            if decoded.startswith(b'\xff\xd8\xff'):
                return "image/jpeg"
            elif decoded.startswith(b'\x89PNG\r\n\x1a\n'):
                return "image/png"
            elif decoded.startswith(b'GIF87a') or decoded.startswith(b'GIF89a'):
                return "image/gif"
            elif decoded.startswith(b'RIFF') and b'WEBP' in decoded[:20]:
                return "image/webp"
            else:
                return "image/jpeg"  # Default fallback
        except Exception:
            return "image/jpeg"  # Default fallback

    def _validate_image_size(self, base64_data: str) -> bool:
        """
        Validate image size according to Groq limits.
        
        Args:
            base64_data (str): Base64 encoded image data
            
        Returns:
            bool: True if image size is within limits
        """
        try:
            import base64
            # Remove data URL prefix if present
            if base64_data.startswith('data:'):
                base64_data = base64_data.split(',')[1]
            
            # Calculate size in MB
            size_bytes = len(base64.b64decode(base64_data))
            size_mb = size_bytes / (1024 * 1024)
            
            # Groq limit for base64 images is 4MB
            MAX_SIZE_MB = 4
            
            if size_mb > MAX_SIZE_MB:
                logger.error(f"Image size {size_mb:.2f}MB exceeds Groq limit of {MAX_SIZE_MB}MB")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Failed to validate image size: {e}")
            return False