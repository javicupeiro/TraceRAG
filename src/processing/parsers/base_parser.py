from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal, Any, Dict, Optional
from pathlib import Path

# Define a strict type for chunks to ensure consistency across the application.
ChunkType = Literal["text", "table", "image"]

@dataclass
class DocumentChunk:
    """
    A standardized data structure representing a single unit of information
    extracted from a document. This class serves as the common return type
    for all parser implementations.
    """
    content: str  # The main content: raw text, Markdown for tables, or a base64 string for images.
    type: ChunkType # The type of the chunk: 'text', 'table', or 'image'.
    source_page: int # The page number in the original document where the chunk was found.
    metadata: Dict[str, Any]  # A dictionary for any additional metadata, such as image captions or table titles.

class BaseParser(ABC):
    """
    Abstract Base Class for all document parsers.
    
    This class defines the essential interface that any concrete parser
    (e.g., PdfParser, DocxParser) must implement. It ensures that the rest of
    the application can interact with any parser in a consistent way.
    """

    @abstractmethod
    def parse(self, file_path: str | Path) -> List[DocumentChunk]:
        """
        Processes a file from a given path and extracts its content into a list of DocumentChunk objects.

        This is the primary method that must be implemented by all subclasses.
        
        Args:
            file_path (str | Path): The path to the document file to be processed.

        Returns:
            List[DocumentChunk]: An ordered list of the chunks extracted from the document.
                                 Returns an empty list if no content could be extracted.
        """
        pass

    def reconstruct_to_markdown(self, file_path: str | Path) -> Optional[str]:
        """
        (Optional) Reconstructs the original document into a single Markdown string.
        
        This method is not required for the core RAG pipeline but can be useful for
        debugging or document conversion purposes. Subclasses may override this
        to provide specific functionality.
        
        Args:
            file_path (str | Path): The path to the document to be processed.

        Returns:
            Optional[str]: A string containing the Markdown content, or None if the
                           functionality is not supported by the parser.
        """
        # By default, this functionality is not implemented.
        # Specific parsers may override this method.
        print(f"Warning: Markdown reconstruction is not implemented for {self.__class__.__name__}")
        return None