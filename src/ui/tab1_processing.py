# src/ui/tab1_processing.py

import streamlit as st
from src.processing.document_processor import DocumentProcessor
from pathlib import Path
import tempfile
import logging

# Get a logger for the current module
logger = logging.getLogger(__name__)

def generate_unique_id() -> str:
    """Generate a unique ID for chunks."""
    import uuid
    return str(uuid.uuid4())

def render_tab1(processor: DocumentProcessor):
    """
    Renders the 'Document Processing' tab in the Streamlit UI.

    This tab handles the document upload and the ingestion pipeline,
    providing granular visual feedback to the user throughout the process.
    Supports both PDF and Markdown files.

    Args:
        processor (DocumentProcessor): An instance of the main DocumentProcessor.
    """
    logger.debug("Rendering Tab 1: Document Processing.")
    
    st.header("Step 1: Process a Document")
    st.markdown("""
    Upload a PDF or Markdown file. The system will parse it into text, table, and image chunks,
    generate a descriptive summary for each, and store them in the databases.
    """)

    # Get language from the global session state
    lang_code = st.session_state.get('language', 'en')
    lang_name = "Español" if lang_code == "es" else "English"
    
    st.info(f"Summaries will be generated in **{lang_name}**. You can change this in the sidebar settings.")

    # File uploader supporting both PDF and Markdown
    uploaded_file = st.file_uploader(
        "Choose a document file", 
        type=["pdf", "md", "markdown"],
        help="Supported formats: PDF (.pdf), Markdown (.md, .markdown)"
    )

    if uploaded_file is not None:
        # Show file info
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        st.success(f"✅ File uploaded: **{uploaded_file.name}** ({file_size_mb:.2f} MB)")
        
        # Check if file type is supported
        if not processor.is_supported_file(uploaded_file.name):
            st.error(f"❌ Unsupported file type: {file_extension}")
            st.info(f"Supported file types: {', '.join(processor.get_supported_file_types())}")
            return

        if st.button(f"🚀 Process Document in {lang_name}", use_container_width=True):
            # Create temporary file with correct extension
            suffix = file_extension if file_extension in ['.pdf', '.md', '.markdown'] else '.tmp'
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = Path(tmp_file.name)
            
            logger.info(f"User uploaded file '{uploaded_file.name}'. Starting processing.")
            
            try:
                # Set the summarizer language before processing
                processor.summarizer.set_language(lang_code)

                with st.status("⚙️ Processing document...", expanded=True) as status:
                    # Phase 1: Parsing
                    status.update(label="Phase 1/3: Parsing document into chunks...", state="running")
                    original_chunks, summaries, summary_vectors = processor.process_document(file_path)
                    
                    if not original_chunks:
                        st.warning("No chunks were extracted from the document. Aborting.")
                        logger.warning(f"Processing aborted for '{uploaded_file.name}': No chunks found.")
                        return
                    
                    total_chunks = len(original_chunks)
                    logger.info(f"Processing complete. Found {total_chunks} chunks.")
                    
                    status.update(label="Processing complete.", state="complete", expanded=False)

                # Display document analysis summary
                counts = {"text": 0, "table": 0, "image": 0, "code": 0}
                for chunk in original_chunks:
                    chunk_type = chunk.type
                    counts[chunk_type] = counts.get(chunk_type, 0) + 1
                
                # Create summary text
                summary_parts = []
                if counts["text"] > 0:
                    summary_parts.append(f"Text chunks: **{counts['text']}**")
                if counts["table"] > 0:
                    summary_parts.append(f"Table chunks: **{counts['table']}**")
                if counts["image"] > 0:
                    summary_parts.append(f"Image chunks: **{counts['image']}**")
                if counts["code"] > 0:
                    summary_parts.append(f"Code chunks: **{counts['code']}**")
                
                st.success(
                    f"**Document analysis complete:**\n" + 
                    "\n".join([f"- {part}" for part in summary_parts])
                )
                st.divider()

                # Persistence with Progress Bar
                st.info("Starting database persistence...")
                progress_bar = st.progress(0, text="Initializing persistence...")
                document_id = uploaded_file.name

                for i, chunk in enumerate(original_chunks):
                    unique_id = generate_unique_id()
                    
                    # Store chunk in SQL database
                    processor.sql_handler.add_chunk(
                        chunk_id=unique_id, 
                        document_id=document_id, 
                        chunk_type=chunk.type,
                        content=chunk.content, 
                        source_page=chunk.source_page,
                        chunk_metadata=chunk.metadata, 
                        summary=summaries[i]
                    )
                    
                    # Store embedding in vector database
                    processor.vector_handler.add_embedding(
                        chunk_id=unique_id, 
                        vector=summary_vectors[i]
                    )
                    
                    # Update progress
                    progress_percentage = (i + 1) / total_chunks
                    progress_text = f"Persisting chunk {i + 1}/{total_chunks}..."
                    progress_bar.progress(progress_percentage, text=progress_text)
                
                progress_bar.progress(1.0, text="✅ Persistence complete!")
                st.balloons()
                st.success(f"🎉 Document '{document_id}' has been successfully processed and stored!")
                logger.info(f"Successfully processed and persisted document '{document_id}'.")

                # Show some statistics
                with st.expander("📊 Processing Statistics", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Chunks", total_chunks)
                        st.metric("Text Chunks", counts["text"])
                    
                    with col2:
                        st.metric("Visual Chunks", counts["table"] + counts["image"])
                        st.metric("Table Chunks", counts["table"])
                    
                    with col3:
                        st.metric("Image Chunks", counts["image"])
                        if counts["code"] > 0:
                            st.metric("Code Chunks", counts["code"])

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                logger.error(f"Failed to process document '{uploaded_file.name}'.", exc_info=True)
                
            finally:
                # Clean up temporary file
                if 'file_path' in locals() and file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Removed temporary file: {file_path}")

    else:
        # Show supported file types when no file is uploaded
        st.info("📁 Please upload a document to begin processing.")
        
        with st.expander("ℹ️ Supported File Types", expanded=False):
            st.markdown("""
            **Supported document formats:**
            - **PDF files** (.pdf) - Extracts text, tables, and images
            - **Markdown files** (.md, .markdown) - Processes text, code blocks, tables, and image references
            
            **Processing capabilities:**
            - Text chunking with configurable size
            - Table extraction (as images for PDF, as markdown for MD)
            - Image processing and base64 conversion
            - Code block detection (Markdown only)
            - Multilingual summarization support
            """)