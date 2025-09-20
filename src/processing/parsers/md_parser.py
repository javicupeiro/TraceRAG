import logging
import re
import base64
import requests
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import mimetypes
from urllib.parse import urlparse, urljoin
import hashlib

# Third-party library imports for Markdown processing with Docling
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker
from docling_core.transforms.chunker import DocChunk
from docling_core.types.doc import DocItem, PictureItem, TableItem
from docling_core.types.doc.labels import DocItemLabel

# Local application imports
from .base_parser import BaseParser, DocumentChunk

# Get a logger for the current module
logger = logging.getLogger(__name__)

class SerializationConfig(Enum):
    """Configuration options for processing different Markdown elements."""
    EXTRACT_TABLES = "extract_tables"
    EXTRACT_IMAGES = "extract_images" 
    EXTRACT_CODE_BLOCKS = "extract_code_blocks"
    PRESERVE_STRUCTURE = "preserve_structure"
    VALIDATE_MARKDOWN = "validate_markdown"

@dataclass
class MarkdownValidationResult:
    """Result of Markdown validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, int]

class MarkdownParser(BaseParser):
    """
    A concrete parser implementation for Markdown documents using the 'Docling' library.
    
    This parser processes Markdown files with advanced capabilities including:
    - Table extraction as separate chunks
    - Image reference processing
    - Code block detection and extraction
    - Configurable serialization options
    - Markdown validation
    """

    def __init__(
        self, 
        chunk_size: int = 256,
        serialization_config: Optional[List[SerializationConfig]] = None,
        validate_markdown: bool = True
    ):
        """
        Initializes the MarkdownParser with advanced processing capabilities.

        Args:
            chunk_size (int): The target maximum number of tokens for each text chunk.
            serialization_config (List[SerializationConfig]): Configuration for element processing.
            validate_markdown (bool): Whether to validate Markdown syntax.
        """
        logger.info(f"Initializing MarkdownParser with chunk_size={chunk_size}")
        
        # Default serialization config
        if serialization_config is None:
            serialization_config = [
                SerializationConfig.EXTRACT_TABLES,
                SerializationConfig.EXTRACT_IMAGES,
                SerializationConfig.EXTRACT_CODE_BLOCKS,
                SerializationConfig.PRESERVE_STRUCTURE,
                SerializationConfig.VALIDATE_MARKDOWN
            ]
        
        self.serialization_config = serialization_config
        self.validate_markdown = validate_markdown
        
        # Configure DocumentConverter for Markdown processing
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.MD]
        )
        
        # Store chunk_size as instance attribute
        self.chunk_size = chunk_size
        self.chunker = HybridChunker(max_tokens_per_chunk=chunk_size)
        
        # Internal state to cache the last parsed document
        self._doc: Optional[DocItem] = None
        self._doc_path: Optional[Path] = None
        self.chunks: List[DocumentChunk] = []
        self.validation_result: Optional[MarkdownValidationResult] = None

    def _validate_markdown_syntax(self, file_path: Path) -> MarkdownValidationResult:
        """
        Validates Markdown syntax and structure.
        
        Args:
            file_path (Path): Path to the Markdown file.
            
        Returns:
            MarkdownValidationResult: Validation results with errors and statistics.
        """
        logger.debug(f"Validating Markdown syntax for: {file_path.name}")
        
        errors = []
        warnings = []
        statistics = {
            "headers": 0,
            "tables": 0,
            "code_blocks": 0,
            "images": 0,
            "links": 0,
            "lines": 0
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                statistics["lines"] = len(lines)
            
            # Check for common Markdown syntax issues
            self._check_headers(content, errors, warnings, statistics)
            self._check_tables(content, errors, warnings, statistics)
            self._check_code_blocks(content, errors, warnings, statistics)
            self._check_images_and_links(content, errors, warnings, statistics)
            
            is_valid = len(errors) == 0
            logger.info(f"Markdown validation complete. Valid: {is_valid}, Errors: {len(errors)}, Warnings: {len(warnings)}")
            
            return MarkdownValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                statistics=statistics
            )
            
        except Exception as e:
            logger.error(f"Error during Markdown validation: {e}")
            errors.append(f"Validation failed: {str(e)}")
            return MarkdownValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                statistics=statistics
            )

    def _check_headers(self, content: str, errors: List[str], warnings: List[str], stats: Dict[str, int]):
        """Check header syntax and hierarchy."""
        lines = content.split('\n')
        prev_level = 0
        
        for i, line in enumerate(lines, 1):
            # ATX headers (# ## ###)
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                stats["headers"] += 1
                
                # Check for skipped header levels
                if level > prev_level + 1:
                    warnings.append(f"Line {i}: Header level {level} skips levels (previous was {prev_level})")
                
                # Check for empty headers
                if not title:
                    errors.append(f"Line {i}: Empty header found")
                
                prev_level = level
            
            # Setext headers (underlined with = or -)
            elif re.match(r'^[=-]+\s*$', line.strip()) and i > 1:
                if lines[i-2].strip():  # Previous line should not be empty
                    stats["headers"] += 1

    def _check_tables(self, content: str, errors: List[str], warnings: List[str], stats: Dict[str, int]):
        """Check table syntax."""
        lines = content.split('\n')
        in_table = False
        table_start = 0
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            
            # Check if line looks like a table row
            if '|' in line and line.startswith('|') and line.endswith('|'):
                if not in_table:
                    in_table = True
                    table_start = i
                    stats["tables"] += 1
                
                # Validate table row syntax
                cells = [cell.strip() for cell in line.split('|')[1:-1]]  # Remove empty first/last
                if not cells:
                    errors.append(f"Line {i}: Empty table row")
                    
            elif in_table and re.match(r'^\|[\s\-\|:]+\|$', line):
                # Table separator row - validate alignment syntax
                pass
            elif in_table and line and '|' not in line:
                # End of table
                in_table = False

    def _check_code_blocks(self, content: str, errors: List[str], warnings: List[str], stats: Dict[str, int]):
        """Check code block syntax."""
        lines = content.split('\n')
        in_fenced_block = False
        fence_pattern = None
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Check for fenced code blocks (``` or ~~~)
            if re.match(r'^(```|~~~)', stripped):
                if not in_fenced_block:
                    in_fenced_block = True
                    fence_pattern = stripped[:3]
                    stats["code_blocks"] += 1
                elif stripped.startswith(fence_pattern):
                    in_fenced_block = False
                    fence_pattern = None
        
        # Check for unclosed code blocks
        if in_fenced_block:
            errors.append("Unclosed fenced code block found")

    def _check_images_and_links(self, content: str, errors: List[str], warnings: List[str], stats: Dict[str, int]):
        """Check image and link syntax."""
        # Images: ![alt](url)
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        images = re.findall(image_pattern, content)
        stats["images"] = len(images)
        
        # Links: [text](url)
        link_pattern = r'(?<!!)\[([^\]]+)\]\(([^)]+)\)'
        links = re.findall(link_pattern, content)
        stats["links"] = len(links)
        
        # Check for malformed image/link syntax
        malformed_pattern = r'!\??\[[^\]]*\]\([^)]*\)'
        malformed = re.findall(malformed_pattern, content)
        if malformed:
            warnings.append(f"Found {len(malformed)} potentially malformed image/link references")

    def _extract_code_blocks_from_content(self, content: str) -> List[Dict[str, Any]]:
        """Extract code blocks from raw Markdown content."""
        code_blocks = []
        lines = content.split('\n')
        in_block = False
        current_block = []
        current_language = None
        block_start = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if re.match(r'^```(.*)$', stripped):
                if not in_block:
                    # Start of code block
                    in_block = True
                    block_start = i + 1
                    current_language = stripped[3:].strip() or None
                    current_block = []
                else:
                    # End of code block
                    in_block = False
                    code_blocks.append({
                        "content": '\n'.join(current_block),
                        "language": current_language,
                        "start_line": block_start,
                        "end_line": i
                    })
                    current_block = []
                    current_language = None
            elif in_block:
                current_block.append(line)
        
        return code_blocks

    def _extract_tables_from_content(self, content: str) -> List[Dict[str, Any]]:
        """Extract table data from raw Markdown content."""
        tables = []
        lines = content.split('\n')
        current_table = []
        in_table = False
        table_start = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if '|' in stripped and stripped.startswith('|') and stripped.endswith('|'):
                if not in_table:
                    in_table = True
                    table_start = i + 1
                    current_table = []
                
                current_table.append(stripped)
                
            elif in_table and re.match(r'^\|[\s\-\|:]+\|$', stripped):
                # Table separator - continue
                current_table.append(stripped)
                
            elif in_table and (not stripped or '|' not in stripped):
                # End of table
                in_table = False
                if current_table:
                    tables.append({
                        "content": '\n'.join(current_table),
                        "start_line": table_start,
                        "end_line": i,
                        "rows": len([row for row in current_table if not re.match(r'^\|[\s\-\|:]+\|$', row)])
                    })
                current_table = []
        
        # Handle table at end of file
        if in_table and current_table:
            tables.append({
                "content": '\n'.join(current_table),
                "start_line": table_start,
                "end_line": len(lines),
                "rows": len([row for row in current_table if not re.match(r'^\|[\s\-\|:]+\|$', row)])
            })
        
        return tables

    def _extract_images_from_content(self, content: str) -> List[Dict[str, Any]]:
        """Extract image references from raw Markdown content."""
        images = []
        pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        
        for match in re.finditer(pattern, content):
            alt_text = match.group(1)
            url = match.group(2)
            start_pos = match.start()
            
            # Count line number
            line_num = content[:start_pos].count('\n') + 1
            
            images.append({
                "alt_text": alt_text,
                "url": url,
                "line": line_num,
                "full_match": match.group(0)
            })
        
        return images

    def _load_and_get_doc(self, file_path: Path) -> DocItem:
        """Loads a Markdown document using Docling with caching."""
        if self._doc is None or self._doc_path != file_path:
            logger.info(f"Loading and processing new Markdown document from: {file_path.name}")
            try:
                conv_res = self.converter.convert(str(file_path))
                self._doc = conv_res.document
                self._doc_path = file_path
                logger.info(f"Markdown document '{file_path.name}' loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load or convert Markdown '{file_path.name}': {e}")
                raise
        else:
            logger.debug(f"Using cached document: {file_path.name}")
        return self._doc

    def parse(self, file_path: str | Path) -> List[DocumentChunk]:
        """
        Processes a Markdown file with advanced extraction capabilities.
        
        Args:
            file_path (str | Path): The path to the Markdown file to be processed.

        Returns:
            List[DocumentChunk]: An ordered list of chunks extracted from the document.
        """
        file_path = Path(file_path)
        logger.info(f"Starting advanced parsing process for Markdown: {file_path.name}")
        
        # Validate file extension
        if file_path.suffix.lower() not in ['.md', '.markdown']:
            logger.warning(f"File {file_path.name} doesn't have a Markdown extension (.md/.markdown)")
        
        # Validate Markdown syntax if configured
        if (SerializationConfig.VALIDATE_MARKDOWN in self.serialization_config and 
            self.validate_markdown):
            self.validation_result = self._validate_markdown_syntax(file_path)
            if not self.validation_result.is_valid:
                logger.warning(f"Markdown validation failed with {len(self.validation_result.errors)} errors")
                for error in self.validation_result.errors:
                    logger.warning(f"Validation error: {error}")

        # Load document with Docling
        doc = self._load_and_get_doc(file_path)
        
        # Read raw content for custom extraction
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        self.chunks = []
        
        # Extract different types of content based on configuration
        if SerializationConfig.EXTRACT_CODE_BLOCKS in self.serialization_config:
            self._process_code_blocks(raw_content, file_path)
        
        if SerializationConfig.EXTRACT_TABLES in self.serialization_config:
            self._process_tables(doc, raw_content, file_path)
            
        if SerializationConfig.EXTRACT_IMAGES in self.serialization_config:
            self._process_images(doc, raw_content, file_path)
        
        # Process text chunks using Docling's chunker
        self._process_text_chunks(doc, file_path)

        logger.info(f"Advanced Markdown parsing complete. Extracted {len(self.chunks)} chunks from {file_path.name}")
        
        # Log chunk type distribution
        chunk_types = {}
        for chunk in self.chunks:
            chunk_types[chunk.type] = chunk_types.get(chunk.type, 0) + 1
        logger.info(f"Chunk distribution: {chunk_types}")
        
        return self.chunks

    def _process_code_blocks(self, content: str, file_path: Path):
        """Process code blocks as separate chunks."""
        logger.debug("Processing code blocks...")
        code_blocks = self._extract_code_blocks_from_content(content)
        
        for i, block in enumerate(code_blocks):
            metadata = {
                "type": "CodeBlock",
                "language": block["language"],
                "start_line": block["start_line"],
                "end_line": block["end_line"],
                "block_index": i,
                "filename": file_path.name
            }
            
            chunk = DocumentChunk(
                content=block["content"],
                type="code",
                source_page=1,  # Markdown files don't have pages
                metadata=metadata
            )
            self.chunks.append(chunk)
        
        logger.info(f"Extracted {len(code_blocks)} code blocks")

    def _process_tables(self, doc: DocItem, content: str, file_path: Path):
        """Process tables as separate chunks."""
        logger.debug("Processing tables...")
        
        # Try to get tables from Docling first (more accurate)
        docling_tables = []
        for item, _ in doc.iterate_items():
            if isinstance(item, TableItem):
                docling_tables.append(item)
        
        if docling_tables:
            logger.debug(f"Found {len(docling_tables)} tables via Docling")
            for i, table_item in enumerate(docling_tables):
                try:
                    # Try to get table as markdown
                    table_content = table_item.export_to_markdown()
                    caption = table_item.caption_text(doc) if hasattr(table_item, 'caption_text') else ""
                    
                    metadata = {
                        "type": "DoclingTable",
                        "table_index": i,
                        "caption": caption,
                        "filename": file_path.name
                    }
                    
                    chunk = DocumentChunk(
                        content=table_content,
                        type="table",
                        source_page=1,
                        metadata=metadata
                    )
                    self.chunks.append(chunk)
                except Exception as e:
                    logger.warning(f"Failed to process Docling table {i}: {e}")
        else:
            # Fallback to manual extraction
            logger.debug("No Docling tables found, using manual extraction")
            manual_tables = self._extract_tables_from_content(content)
            
            for i, table in enumerate(manual_tables):
                metadata = {
                    "type": "ManualTable",
                    "table_index": i,
                    "start_line": table["start_line"],
                    "end_line": table["end_line"],
                    "rows": table["rows"],
                    "filename": file_path.name
                }
                
                chunk = DocumentChunk(
                    content=table["content"],
                    type="table",
                    source_page=1,
                    metadata=metadata
                )
                self.chunks.append(chunk)
            
            logger.info(f"Extracted {len(manual_tables)} tables manually")

    def _process_images(self, 
                        doc: DocItem, 
                        content: str, 
                        file_path: Path):
        """Process image references as separate chunks."""
        logger.debug("Processing images...")
        
        # Extract image references from content
        images = self._extract_images_from_content(content)
        
        for i, img in enumerate(images):
            metadata = {
                "type": "ImageReference",
                "alt_text": img["alt_text"],
                "url": img["url"],
                "line": img["line"],
                "image_index": i,
                "filename": file_path.name
            }
            
            chunk_content = img["full_match"]
            
            # Try to convert to base64 if it is a local image
            if not img["url"].startswith(('http://', 'https://')):
                base64_result = self._convert_local_image_to_base64(img["url"], file_path)
                
                if base64_result:
                    metadata.update({
                        "base64_data": base64_result['base64_data'],
                        "mime_type": base64_result['mime_type'],
                        "file_size_bytes": base64_result['file_size'],
                        "resolved_path": base64_result['resolved_path'],
                        "is_local": True,
                        "base64_available": True
                    })
                    logger.info(f"✅ Local image converted: {img['url']}")
                else:
                    metadata.update({
                        "is_local": True,
                        "base64_available": False,
                        "error": "Could not convert to base64"
                    })
                    logger.warning(f"❌ Could not convert: {img['url']}")
            else:
                metadata.update({
                    "is_local": False,
                    "base64_available": False
                })
            
            chunk = DocumentChunk(
                content=chunk_content,
                type="image",
                source_page=1,
                metadata=metadata
            )
            self.chunks.append(chunk)
        
        local_images = sum(1 for img in images if not img["url"].startswith(('http://', 'https://')))
        converted_images = sum(1 for chunk in self.chunks 
                              if chunk.type == "image" and 
                              chunk.metadata.get("base64_available", False))
        
        logger.info(f"Extracted {len(images)} image references ({local_images} local, {converted_images} converted to base64)")


    def _process_text_chunks(self, doc: DocItem, file_path: Path):
        """Process regular text content using Docling's chunker."""
        logger.debug("Processing text chunks...")
        
        try:
            text_doc_chunks = list(self.chunker.chunk(doc))
            processed_chunks = 0
            
            for chunk in text_doc_chunks:
                if isinstance(chunk, DocChunk):
                    chunk_text = chunk.text.strip() if chunk.text else ""
                    
                    if not chunk_text:
                        continue
                    
                    metadata = {
                        "type": "TextChunk",
                        "chunk_index": processed_chunks,
                        "filename": file_path.name
                    }
                    
                    # Add headings information if available
                    if chunk.meta and hasattr(chunk.meta, 'headings') and chunk.meta.headings:
                        metadata["headings"] = chunk.meta.headings
                    
                    # Add origin information if available
                    if chunk.meta and hasattr(chunk.meta, 'origin'):
                        metadata["origin"] = {
                            "filename": chunk.meta.origin.filename if hasattr(chunk.meta.origin, 'filename') else file_path.name,
                            "mimetype": chunk.meta.origin.mimetype if hasattr(chunk.meta.origin, 'mimetype') else "text/markdown"
                        }

                    new_chunk = DocumentChunk(
                        content=chunk_text,
                        type="text",
                        source_page=1,  # Markdown doesn't have pages
                        metadata=metadata
                    )
                    
                    self.chunks.append(new_chunk)
                    processed_chunks += 1
            
            logger.info(f"Processed {processed_chunks} text chunks")
            
        except Exception as e:
            logger.error(f"Error during text chunk processing: {e}", exc_info=True)
            raise

    def get_validation_result(self) -> Optional[MarkdownValidationResult]:
        """Get the validation result from the last parse operation."""
        return self.validation_result

    def reconstruct_to_markdown(self, file_path: str | Path) -> Optional[str]:
        """
        Reconstructs the original Markdown document as a single string.
        
        Args:
            file_path (str | Path): The path to the Markdown document to be processed.

        Returns:
            Optional[str]: The reconstructed Markdown content, or None if processing fails.
        """
        logger.info(f"Reconstructing Markdown document: {file_path}")
        file_path = Path(file_path)
        
        try:
            doc = self._load_and_get_doc(file_path)
            markdown_content = doc.export_to_markdown()
            logger.info(f"Successfully reconstructed Markdown for {file_path.name}")
            return markdown_content
        except Exception as e:
            logger.error(f"Failed to reconstruct Markdown for {file_path.name}: {e}", exc_info=True)
            return None

    def get_document_structure(self, file_path: str | Path) -> Optional[Dict[str, Any]]:
        """
        Extracts comprehensive structural information from the Markdown document.
        
        Args:
            file_path (str | Path): The path to the Markdown document.

        Returns:
            Optional[dict]: Dictionary containing structural information about the document.
        """
        logger.debug(f"Extracting document structure for: {file_path}")
        file_path = Path(file_path)
        
        try:
            doc = self._load_and_get_doc(file_path)
            
            # Read raw content for additional analysis
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            
            structure_info = {
                "filename": file_path.name,
                "total_chunks": len(self.chunks),
                "document_name": doc.name if hasattr(doc, 'name') else file_path.stem,
                "serialization_config": [config.value for config in self.serialization_config],
            }
            
            # Add validation results if available
            if self.validation_result:
                structure_info["validation"] = {
                    "is_valid": self.validation_result.is_valid,
                    "error_count": len(self.validation_result.errors),
                    "warning_count": len(self.validation_result.warnings),
                    "statistics": self.validation_result.statistics
                }
            
            # Add chunk type distribution
            chunk_types = {}
            for chunk in self.chunks:
                chunk_types[chunk.type] = chunk_types.get(chunk.type, 0) + 1
            structure_info["chunk_distribution"] = chunk_types
            
            # Add content analysis
            structure_info["content_analysis"] = {
                "file_size_bytes": len(raw_content.encode('utf-8')),
                "line_count": len(raw_content.split('\n')),
                "character_count": len(raw_content),
                "word_count": len(raw_content.split())
            }
            
            logger.debug(f"Extracted comprehensive structure info for {file_path.name}")
            return structure_info
            
        except Exception as e:
            logger.error(f"Failed to extract structure for {file_path.name}: {e}", exc_info=True)
            return None
        
    def _convert_local_image_to_base64(self, image_path: str, markdown_file_path: Path) -> Optional[Dict[str, str]]:
        """
        Convert a local image to base64.

        Args:
            image_path (str): Path of the image from the markdown
            markdown_file_path (Path): Path of the markdown file

        Returns:
            Optional[Dict[str, str]]: Dictionary with base64_data and mime_type, or None if it fails
        """
        try:
            # Resolve relative path
            if not Path(image_path).is_absolute():
                # Try relative to the markdown file
                full_path = markdown_file_path.parent / image_path
                if not full_path.exists():
                    # Try relative to the current directory
                    full_path = Path(image_path)
            else:
                full_path = Path(image_path)

            # Check that the file exists
            if not full_path.exists():
                logger.warning(f"Image not found: {image_path}")
                return None

            # Read file
            with open(full_path, 'rb') as f:
                image_data = f.read()

            # Convert to base64
            base64_data = base64.b64encode(image_data).decode('utf-8')

            # Detect MIME type
            mime_type, _ = mimetypes.guess_type(str(full_path))
            if not mime_type or not mime_type.startswith('image/'):
                # Fallback based on extension
                ext = full_path.suffix.lower()
                mime_type = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg', 
                    '.png': 'image/png',
                    '.gif': 'image/gif',
                    '.webp': 'image/webp',
                    '.svg': 'image/svg+xml'
                }.get(ext, 'image/jpeg')

            logger.info(f"Image converted to base64: {full_path.name} ({len(image_data)} bytes)")

            return {
                'base64_data': base64_data,
                'mime_type': mime_type,
                'file_size': len(image_data),
                'resolved_path': str(full_path)
            }

        except Exception as e:
            logger.error(f"Error converting image to base64 {image_path}: {e}")
            return None

