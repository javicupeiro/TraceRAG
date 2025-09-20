import logging
from pathlib import Path
from typing import List, Tuple
import mimetypes

# Local application imports
from src.processing.parsers.base_parser import BaseParser, DocumentChunk
from src.database.sql_handler import SQLHandler
from src.database.vector_handler import VectorHandler
from src.processing.multimodal_summarizer import MultimodalSummarizer
from src.core.embedder import Embedder

# Get a logger for the current module
logger = logging.getLogger(__name__)

def generate_unique_id() -> str:
    """Generate a unique ID for chunks."""
    import uuid
    return str(uuid.uuid4())

class DocumentProcessor:
    """
    Orchestrates the entire document ingestion pipeline.
    
    This class coordinates the parsing, summarization, embedding, and storage
    of document chunks, acting as the central component for the RAG ingestion process.
    Supports both PDF and Markdown files.
    """
    def __init__(
        self,
        summarizer: MultimodalSummarizer,
        embedder: Embedder,
        sql_handler: SQLHandler,
        vector_handler: VectorHandler,
        chunk_size: int = 256,
        image_resolution_scale: float = 2.0
    ):
        """
        Initializes the DocumentProcessor with all necessary components.

        Args:
            summarizer: The summarizer instance for creating text descriptions.
            embedder: The embedding model instance.
            sql_handler (SQLHandler): The handler for the relational database.
            vector_handler (VectorHandler): The handler for the vector database.
            chunk_size (int): The target maximum number of tokens for each text chunk.
            image_resolution_scale (float): A scaling factor for PDF image resolution.
        """
        self.summarizer = summarizer
        self.embedder = embedder
        self.sql_handler = sql_handler
        self.vector_handler = vector_handler
        self.chunk_size = chunk_size
        self.image_resolution_scale = image_resolution_scale
        
        # Initialize parsers
        self.pdf_parser = PdfParser(
            chunk_size=chunk_size, 
            image_resolution_scale=image_resolution_scale
        )
        self.md_parser = MarkdownParser(chunk_size=chunk_size)
        
        logger.info("DocumentProcessor initialized with PDF and Markdown support.")

    def _get_parser_for_file(self, file_path: Path) -> BaseParser:
        """
        Determines the appropriate parser based on file extension.
        
        Args:
            file_path (Path): The path to the document file.
            
        Returns:
            BaseParser: The appropriate parser instance.
            
        Raises:
            ValueError: If the file type is not supported.
        """
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.pdf':
            logger.debug(f"Using PDF parser for: {file_path.name}")
            return self.pdf_parser
        elif file_extension in ['.md', '.markdown']:
            logger.debug(f"Using Markdown parser for: {file_path.name}")
            return self.md_parser
        else:
            # Try to determine by MIME type as fallback
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type == 'application/pdf':
                logger.debug(f"Using PDF parser for: {file_path.name} (detected by MIME type)")
                return self.pdf_parser
            elif mime_type == 'text/markdown':
                logger.debug(f"Using Markdown parser for: {file_path.name} (detected by MIME type)")
                return self.md_parser
            else:
                supported_extensions = ['.pdf', '.md', '.markdown']
                raise ValueError(
                    f"Unsupported file type: {file_extension}. "
                    f"Supported extensions: {supported_extensions}"
                )

    def process_document(self, file_path: str | Path) -> Tuple[List[DocumentChunk], List[str], List[List[float]]]:
        """
        Executes the processing steps (parse, summarize, embed) without persisting the data.
        
        This method is designed to be called by the UI to allow for more granular
        control over the pipeline, such as displaying progress.

        Args:
            file_path (str | Path): The path to the document to be processed.

        Returns:
            A tuple containing:
            - List[DocumentChunk]: The original chunks extracted from the document.
            - List[str]: The generated summaries for each chunk.
            - List[List[float]]: The generated vector embeddings for each summary.
        """
        file_path = Path(file_path)
        document_id = file_path.name
        logger.info(f"Starting document processing pipeline for: {document_id}")

        # 1. Get the appropriate parser and parse document into chunks
        logger.info("Step 1/3: Parsing document...")
        try:
            parser = self._get_parser_for_file(file_path)
            original_chunks: List[DocumentChunk] = parser.parse(file_path)
        except ValueError as e:
            logger.error(f"Parser selection failed for {document_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Document parsing failed for {document_id}: {e}")
            raise
        
        if not original_chunks:
            logger.warning(f"No chunks were extracted from {document_id}. Aborting processing.")
            return [], [], []
        logger.info(f"Successfully parsed {len(original_chunks)} chunks.")

        # 2. Generate summaries for each chunk
        logger.info("Step 2/3: Generating summaries for each chunk...")
        try:
            summaries = [self.summarizer.summarize_chunk(chunk) for chunk in original_chunks]
            logger.info("Summaries generated successfully.")
        except Exception as e:
            logger.error(f"An error occurred during summarization: {e}", exc_info=True)
            raise

        # 3. Create embeddings for the summaries
        logger.info("Step 3/3: Creating embeddings for summaries...")
        try:
            summary_vectors = self.embedder.embed(summaries)
            logger.info("Embeddings created successfully.")
        except Exception as e:
            logger.error(f"An error occurred during embedding creation: {e}", exc_info=True)
            raise
        
        logger.info(f"Successfully processed {len(original_chunks)} chunks from {document_id}. Data is ready for persistence.")
        return original_chunks, summaries, summary_vectors

    def process_and_persist(self, file_path: str | Path):
        """
        Executes the full RAG ingestion pipeline for a single document, including persistence.
        This is a convenience method for non-UI or scripted use.
        """
        logger.info(f"Starting full process-and-persist pipeline for: {file_path}")
        original_chunks, summaries, summary_vectors = self.process_document(file_path)

        if not original_chunks:
            return

        logger.info(f"Persisting {len(original_chunks)} chunks to databases...")
        document_id = Path(file_path).name
        
        for i, chunk in enumerate(original_chunks):
            unique_id = generate_unique_id()
            try:
                # Persist the original chunk data to the SQL database
                self.sql_handler.add_chunk(
                    chunk_id=unique_id, document_id=document_id, chunk_type=chunk.type,
                    content=chunk.content, source_page=chunk.source_page,
                    chunk_metadata=chunk.metadata, summary=summaries[i]
                )
                
                # Persist the summary's vector to the vector database
                self.vector_handler.add_embedding(
                    chunk_id=unique_id, vector=summary_vectors[i]
                )
            except Exception as e:
                logger.error(f"Failed to persist chunk {i+1}/{len(original_chunks)} (ID: {unique_id}): {e}", exc_info=True)
                continue

        logger.info(f"Successfully processed and persisted all chunks from {document_id}.")

    def get_supported_file_types(self) -> List[str]:
        """
        Returns a list of supported file extensions.
        
        Returns:
            List[str]: List of supported file extensions.
        """
        return ['.pdf', '.md', '.markdown']

    def is_supported_file(self, file_path: str | Path) -> bool:
        """
        Checks if a file type is supported.
        
        Args:
            file_path (str | Path): The path to the file to check.
            
        Returns:
            bool: True if the file type is supported, False otherwise.
        """
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.get_supported_file_types()


