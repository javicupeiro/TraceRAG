import base64
import io
import logging
from pathlib import Path
from typing import List, Optional, Any
from PIL.Image import Image

# Third-party library imports for PDF processing
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import DocItem, PictureItem, TableItem
from docling.chunking import HybridChunker
from docling_core.transforms.chunker import DocChunk 

# Local application imports
from .base_parser import BaseParser, DocumentChunk

# Get a logger for the current module
logger = logging.getLogger(__name__)

class PdfParser(BaseParser):
    """
    A concrete parser implementation for PDF documents using the 'Docling' library.
    
    This parser is capable of extracting not only text but also visual elements
    like tables and images, processing them into a unified list of DocumentChunk objects.
    """

    def __init__(self, chunk_size: int = 256, image_resolution_scale: float = 2.0):
        """
        Initializes the PdfParser with Docling and its chunking strategy.

        Args:
            chunk_size (int): The target maximum number of tokens for each text chunk.
            image_resolution_scale (float): A scaling factor for the resolution of
                                            extracted images. Higher values result in
                                            larger, more detailed images.
        """
        logger.info(f"Initializing PdfParser with chunk_size={chunk_size} and image_scale={image_resolution_scale}")
        
        # Configure Docling to extract images and generate high-quality renderings
        pipeline_options = PdfPipelineOptions(
            images_scale=image_resolution_scale,
            generate_page_images=True,
            generate_picture_images=True
        )
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        # The chunk_size must be stored as an instance attribute to be accessible in other methods.
        self.chunk_size = chunk_size
        self.chunker = HybridChunker(max_tokens_per_chunk=chunk_size)
        
        # Internal state to cache the last parsed document, avoiding reprocessing.
        self._doc: Optional[DocItem] = None
        self._doc_path: Optional[Path] = None
        self.chunks: List[DocumentChunk] = []

    def _image_to_base64(self, image: Image) -> str:
        """Encodes a PIL Image object into a base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def _load_and_get_doc(self, file_path: Path) -> DocItem:
        """
        Loads a document using Docling, caching the result to avoid redundant loads.
        """
        if self._doc is None or self._doc_path != file_path:
            logger.info(f"Loading and processing new document from: {file_path.name}")
            try:
                conv_res = self.converter.convert(str(file_path))
                self._doc = conv_res.document
                self._doc_path = file_path
                logger.info(f"Document '{file_path.name}' loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load or convert PDF '{file_path.name}': {e}")
                raise
        else:
            logger.debug(f"Using cached document: {file_path.name}")
        return self._doc

    def parse(self, file_path: str | Path) -> List[DocumentChunk]:
        """
        Processes a PDF file, extracting text, tables, and images into a structured list of chunks.
        """
        file_path = Path(file_path)
        logger.info(f"Starting parsing process for PDF: {file_path.name}")
        doc = self._load_and_get_doc(file_path)

        # 1. Use the HybridChunker for sophisticated text chunking.
        logger.debug(f"Applying HybridChunker to extract text chunks (max_tokens: {self.chunk_size}).")
        text_doc_chunks = list(self.chunker.chunk(doc))
        
        # 2. Explicitly iterate through the original document to find visual elements.
        logger.debug("Extracting visual elements (tables and pictures) from the document.")
        visual_elements = [item for item, _ in doc.iterate_items() if isinstance(item, (TableItem, PictureItem))]
        
        # 3. Combine both lists for unified processing.
        all_elements_to_process: List[Any] = text_doc_chunks + visual_elements
        logger.info(f"Found {len(text_doc_chunks)} text chunks and {len(visual_elements)} visual elements.")
                
        # 4. Convert all found elements into the standardized DocumentChunk format.
        self.chunks = []
        counts = {"text": 0, "table": 0, "image": 0}

        for item in all_elements_to_process:
            new_chunk = None
            try:
                if isinstance(item, DocChunk):  # Processed text chunk
                    if item.text and item.text.strip():
                        page_num = item.meta.doc_items[0].prov[0].page_no if (item.meta and item.meta.doc_items) else 0
                        new_chunk = DocumentChunk(content=item.text, type="text", source_page=page_num, metadata={"type": "HybridChunk"})
                        counts["text"] += 1
                
                elif isinstance(item, TableItem):
                    page_num = item.prov[0].page_no if item.prov else 0
                    image = item.get_image(doc=doc)
                    image_b64 = self._image_to_base64(image)
                    caption = item.caption_text(doc=doc)
                    new_chunk = DocumentChunk(content=image_b64, type="table", source_page=page_num, metadata={"caption": caption or ""})
                    counts["table"] += 1
                
                elif isinstance(item, PictureItem):
                    page_num = item.prov[0].page_no if item.prov else 0
                    image = item.get_image(doc=doc)
                    image_b64 = self._image_to_base64(image)
                    caption = item.caption_text(doc=doc)
                    new_chunk = DocumentChunk(content=image_b64, type="image", source_page=page_num, metadata={"caption": caption or ""})
                    counts["image"] += 1
                
                if new_chunk:
                    self.chunks.append(new_chunk)
            except Exception as e:
                logger.error(f"Failed to process an item from the document: {e}", exc_info=True)

        logger.info(f"PDF parsing complete. Extracted a total of {len(self.chunks)} chunks.")
        logger.info(f"Extraction summary: {counts['text']} text, {counts['table']} table, {counts['image']} image chunks.")
        return self.chunks