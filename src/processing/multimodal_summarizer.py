import logging
from pathlib import Path
from typing import Dict
import base64

from llm_provider.base_llm_provider import BaseLLMProvider, LLMChunk
from processing.parsers.base_parser import DocumentChunk

# Get a logger for the current module
logger = logging.getLogger(__name__)

class MultimodalSummarizer:
    """
    Generates summaries for DocumentChunk objects using LLM providers.
    
    This class is responsible for selecting the correct prompt based on the chunk type
    (text, image, table) and language, formatting it, and calling the LLM to get a
    descriptive summary. It supports dynamic language switching.
    """
    def __init__(self, llm_provider: BaseLLMProvider, base_prompt_dir: str | Path):
        """
        Initializes the MultimodalSummarizer.

        Args:
            llm_provider (BaseLLMProvider): An instance of a LLM provider for communication.
            base_prompt_dir (str | Path): The base directory containing language-specific subfolders
                                          for prompts (e.g., 'prompts/en', 'prompts/es').
        """
        self.llm_provider = llm_provider
        self.base_prompt_dir = Path(base_prompt_dir)
        
        # Standard filenames for prompts, expected inside each language folder.
        self.prompt_filenames = {
            'text': 'summarize_text.txt',
            'table': 'summarize_table.txt',
            'image': 'summarize_img.txt'
        }
        
        self.prompt_templates: Dict[str, Path] = {}
        self.loaded_prompts: Dict[str, str] = {} # Cache for prompt file content

        logger.info("MultimodalSummarizer initialized.")
        # Set a default language on startup to ensure paths are populated.
        self.set_language('en')

    def set_language(self, lang_code: str):
        """
        Sets the active language for summarization prompts.

        This method rebuilds the paths to the prompt files based on the selected
        language code and clears the prompt cache.

        Args:
            lang_code (str): The language code, which must match a subfolder name
                             (e.g., 'es', 'en').
        """
        logger.info(f"Setting summarizer language to: {lang_code}")
        self.loaded_prompts.clear()
        
        lang_dir = self.base_prompt_dir / lang_code
        if not lang_dir.is_dir():
            logger.error(f"Language directory not found: {lang_dir}. Falling back to 'en'.")
            lang_dir = self.base_prompt_dir / 'en'

        for prompt_type, filename in self.prompt_filenames.items():
            self.prompt_templates[prompt_type] = lang_dir / filename
        logger.debug(f"Prompt paths updated for language '{lang_code}'.")

    def _load_prompt(self, prompt_type: str) -> str:
        """
        Loads a prompt template from a file, using a simple cache to avoid re-reading.

        Args:
            prompt_type (str): The type of prompt to load ('text', 'image', 'table').

        Returns:
            str: The content of the prompt template file.
        """
        if prompt_type in self.loaded_prompts:
            logger.debug(f"Using cached prompt for type '{prompt_type}'.")
            return self.loaded_prompts[prompt_type]

        prompt_path = self.prompt_templates.get(prompt_type)
        if not prompt_path or not prompt_path.exists():
            logger.error(f"Prompt template file not found for type '{prompt_type}' at path: {prompt_path}")
            return "Summarize the following content as accurately as possible."

        try:
            logger.debug(f"Loading prompt for '{prompt_type}' from {prompt_path}.")
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_content = f.read()
            self.loaded_prompts[prompt_type] = prompt_content
            return prompt_content
        except Exception as e:
            logger.error(f"Error reading prompt file {prompt_path}: {e}")
            return "Summarize the following content."

    def _is_valid_base64(self, data: str) -> bool:
        """
        Validates if the provided string is valid base64 data.

        Args:
            data (str): Base64 string to validate

        Returns:
            bool: True if valid base64 data
        """
        try:
            base64.b64decode(data)
            return True
        except Exception:
            return False

    def summarize_chunk(self, chunk: DocumentChunk) -> str:
        """
        Generates a summary for a single DocumentChunk.

        Args:
            chunk (DocumentChunk): The document chunk to be summarized.

        Returns:
            str: The generated summary as a string.
        """
        logger.info(f"Generating summary for chunk of type '{chunk.type}' from page {chunk.source_page}.")
        
        prompt_template = self._load_prompt(chunk.type)

        # Format the prompt based on the chunk type
        if chunk.type == 'text':
            final_prompt = prompt_template.format(text_content=chunk.content)
            llm_chunk = LLMChunk(content=chunk.content, type="text", metadata=chunk.metadata)
        
        elif chunk.type == 'table':
            # For tables, check if content is base64 or markdown table
            if self._is_valid_base64(chunk.content):
                # Table as image (base64)
                caption = chunk.metadata.get("caption", "").strip()
                caption_context = f"Context from caption: '{caption}'" if caption else "No caption was provided."
                final_prompt = prompt_template.format(caption_text=caption_context)
                llm_chunk = LLMChunk(content=chunk.content, type="image", metadata=chunk.metadata)
            else:
                # Table as markdown text
                caption = chunk.metadata.get("caption", "").strip()
                caption_context = f"Context from caption: '{caption}'" if caption else "No caption was provided."
                final_prompt = prompt_template.format(caption_text=caption_context)
                llm_chunk = LLMChunk(content=chunk.content, type="table", metadata=chunk.metadata)
        
        elif chunk.type == 'image':
            caption = chunk.metadata.get("caption", "").strip()
            caption_context = f"Context from caption: '{caption}'" if caption else "No caption was provided."
            final_prompt = prompt_template.format(caption_text=caption_context)
            llm_chunk = LLMChunk(content=chunk.content, type="image", metadata=chunk.metadata)
        
        else:
            logger.warning(f"Unknown chunk type '{chunk.type}', treating as text")
            final_prompt = prompt_template.format(text_content=chunk.content)
            llm_chunk = LLMChunk(content=chunk.content, type="text", metadata=chunk.metadata)

        try:
            response = self.llm_provider.generate_summary(llm_chunk, final_prompt)
            summary = response.content
            logger.info(f"Successfully generated summary for chunk (type: '{chunk.type}').")
            return summary
        except Exception as e:
            logger.error(f"Error generating summary for chunk (type: '{chunk.type}'): {e}")
            return f"Error generating summary: {str(e)}"