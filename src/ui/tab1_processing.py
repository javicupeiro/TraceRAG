# src/ui/tab1_processing.py

import streamlit as st
from processing.document_processor import DocumentProcessor
from llm_provider.llm_factory import LLMProviderFactory
from processing.multimodal_summarizer import MultimodalSummarizer
from core.config_manager import get_document_processing_config
from pathlib import Path
import tempfile
import logging
import os
from typing import List

# Get a logger for the current module
logger = logging.getLogger(__name__)

def generate_unique_id() -> str:
    """Generate a unique ID for chunks."""
    import uuid
    return str(uuid.uuid4())

def find_documents_in_folder(folder_path: str, supported_extensions: List[str]) -> List[Path]:
    """
    Find all supported documents in a folder and its subfolders.
    
    Args:
        folder_path (str): Path to the folder to search
        supported_extensions (List[str]): List of supported file extensions
        
    Returns:
        List[Path]: List of found document files
    """
    folder = Path(folder_path)
    if not folder.exists():
        return []
    
    documents = []
    for ext in supported_extensions:
        # Search recursively for files with the extension
        documents.extend(folder.rglob(f"*{ext}"))
    
    return sorted(documents)

def process_multiple_documents(documents: List[Path], processor: DocumentProcessor, 
                             llm_provider, base_prompt_dir: str, lang_code: str,
                             selected_provider: str):
    """
    Process multiple documents with progress tracking.
    
    Args:
        documents (List[Path]): List of document paths to process
        processor (DocumentProcessor): Document processor instance
        llm_provider: LLM provider instance
        base_prompt_dir (str): Base prompt directory
        lang_code (str): Language code
        selected_provider (str): Selected provider name
    """
    total_documents = len(documents)
    total_chunks_processed = 0
    
    # Create progress containers
    overall_progress = st.progress(0, text="Initializing batch processing...")
    document_status = st.empty()
    processing_stats = st.empty()
    
    # Create summarizer with selected provider and language
    summarizer = MultimodalSummarizer(llm_provider, base_prompt_dir)
    summarizer.set_language(lang_code)
    
    # Update processor's summarizer
    original_summarizer = processor.summarizer
    processor.summarizer = summarizer
    
    successful_docs = 0
    failed_docs = []
    
    try:
        for doc_idx, doc_path in enumerate(documents):
            document_status.info(f"Processing: **{doc_path.name}** ({doc_idx + 1}/{total_documents})")
            
            try:
                # Process document
                original_chunks, summaries, summary_vectors = processor.process_document(doc_path)
                
                if not original_chunks:
                    st.warning(f"No chunks extracted from {doc_path.name}")
                    failed_docs.append(f"{doc_path.name} (no chunks found)")
                    continue
                
                # Persist chunks
                document_id = doc_path.name
                for i, chunk in enumerate(original_chunks):
                    unique_id = generate_unique_id()
                    
                    processor.sql_handler.add_chunk(
                        chunk_id=unique_id, 
                        document_id=document_id, 
                        chunk_type=chunk.type,
                        content=chunk.content, 
                        source_page=chunk.source_page,
                        chunk_metadata=chunk.metadata, 
                        summary=summaries[i]
                    )
                    
                    processor.vector_handler.add_embedding(
                        chunk_id=unique_id, 
                        vector=summary_vectors[i]
                    )
                
                total_chunks_processed += len(original_chunks)
                successful_docs += 1
                
                # Update progress
                progress = (doc_idx + 1) / total_documents
                overall_progress.progress(progress, 
                    text=f"Processed {doc_idx + 1}/{total_documents} documents")
                
                # Update stats
                processing_stats.success(
                    f"✅ **Progress:** {successful_docs} documents processed successfully\n"
                    f"📄 **Total chunks:** {total_chunks_processed}\n"
                    f"🤖 **Provider:** {selected_provider}"
                )
                
            except Exception as e:
                logger.error(f"Failed to process {doc_path.name}: {e}")
                failed_docs.append(f"{doc_path.name} ({str(e)})")
                continue
        
        # Final results
        overall_progress.progress(1.0, text="✅ Batch processing complete!")
        
        if successful_docs > 0:
            st.balloons()
            st.success(
                f"🎉 **Batch processing complete!**\n\n"
                f"✅ **Successfully processed:** {successful_docs} documents\n"
                f"📄 **Total chunks created:** {total_chunks_processed}\n"
                f"🤖 **Provider used:** {selected_provider}"
            )
        
        if failed_docs:
            st.error(f"❌ **Failed to process {len(failed_docs)} documents:**")
            for doc in failed_docs:
                st.text(f"• {doc}")
    
    finally:
        # Restore original summarizer
        processor.summarizer = original_summarizer

def render_tab1(processor: DocumentProcessor, base_prompt_dir: str):
    """
    Renders the 'Document Processing' tab in the Streamlit UI.

    This tab handles the document upload and the ingestion pipeline,
    providing granular visual feedback to the user throughout the process.
    Supports both PDF and Markdown files with configurable LLM provider and language.
    Now includes batch processing from a specified folder.

    Args:
        processor (DocumentProcessor): An instance of the main DocumentProcessor.
        base_prompt_dir (str): Base directory for prompt templates.
    """
    logger.debug("Rendering Tab 1: Document Processing.")
    
    st.header("📄 Document Processing")
    st.markdown("""
    Process documents individually or in batch. The system will parse them into text, table, and image chunks,
    generate a descriptive summary for each, and store them in the databases.
    """)

    # Configuration Section
    st.subheader("🔧 Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Language selector
        language_options = {
            'English': 'en',
            'Español': 'es'
        }
        selected_language = st.selectbox(
            "🌍 Summary Language",
            options=list(language_options.keys()),
            index=0,
            help="Language for generating summaries"
        )
        lang_code = language_options[selected_language]
    
    with col2:
        # LLM Provider selector
        provider_options = {
            'Groq (llama-4-scout-17b-16e-instruct)': 'groq',
            'Google Gemini (gemini-1.5-flash)': 'gemini'
        }
        selected_provider = st.selectbox(
            "🤖 LLM Provider",
            options=list(provider_options.keys()),
            index=0,
            help="Choose the language model provider for summarization"
        )
        provider_name = provider_options[selected_provider]
    
    # Show current configuration
    st.info(f"📋 **Current Setup:** {selected_provider} | {selected_language} summaries")
    
    st.divider()

    # Processing mode selector
    st.subheader("📂 Processing Mode")
    
    processing_mode = st.radio(
        "Choose processing mode:",
        options=["Single File Upload", "Batch Process Folder"],
        horizontal=True
    )

    if processing_mode == "Single File Upload":
        render_single_file_processing(processor, base_prompt_dir, lang_code, provider_name, selected_provider, selected_language)
    else:
        render_batch_folder_processing(processor, base_prompt_dir, lang_code, provider_name, selected_provider, selected_language)

def render_single_file_processing(processor, base_prompt_dir, lang_code, provider_name, selected_provider, selected_language):
    """Render single file upload processing interface."""
    
    st.subheader("📤 Single File Upload")
    
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

        if st.button(f"🚀 Process Document", use_container_width=True):
            # Create temporary file with correct extension
            suffix = file_extension if file_extension in ['.pdf', '.md', '.markdown'] else '.tmp'
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = Path(tmp_file.name)
            
            logger.info(f"User uploaded file '{uploaded_file.name}'. Processing with {provider_name} in {lang_code}")
            
            try:
                # Create LLM provider based on user selection
                with st.spinner(f"🔄 Initializing {selected_provider}..."):
                    try:
                        llm_provider = LLMProviderFactory.create_provider(provider_name)
                        logger.info(f"Successfully created {provider_name} provider")
                    except Exception as e:
                        st.error(f"❌ Failed to initialize {selected_provider}: {str(e)}")
                        st.error("Please check your API keys in environment variables")
                        logger.error(f"Failed to create {provider_name} provider: {e}")
                        return
                
                # Create new summarizer with selected provider and language
                summarizer = MultimodalSummarizer(llm_provider, base_prompt_dir)
                summarizer.set_language(lang_code)
                
                # Update processor's summarizer
                original_summarizer = processor.summarizer
                processor.summarizer = summarizer

                with st.status("⚙️ Processing document...", expanded=True) as status:
                    # Phase 1: Parsing
                    status.update(label="Phase 1/3: Parsing document into chunks...", state="running")
                    st.write("📄 Extracting content from document...")
                    
                    # Phase 2: Summarization
                    status.update(label="Phase 2/3: Generating summaries...", state="running")
                    st.write(f"🤖 Using {selected_provider} for {selected_language} summaries...")
                    
                    # Phase 3: Embedding
                    status.update(label="Phase 3/3: Creating embeddings...", state="running")
                    st.write("🔢 Converting summaries to vector embeddings...")
                    
                    # Process document
                    original_chunks, summaries, summary_vectors = processor.process_document(file_path)
                    
                    if not original_chunks:
                        st.warning("No chunks were extracted from the document. Aborting.")
                        logger.warning(f"Processing aborted for '{uploaded_file.name}': No chunks found.")
                        return
                    
                    total_chunks = len(original_chunks)
                    logger.info(f"Processing complete. Found {total_chunks} chunks.")
                    
                    status.update(label="✅ Processing complete.", state="complete", expanded=False)

                # Restore original summarizer
                processor.summarizer = original_summarizer

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
                    col1, col2, col3, col4 = st.columns(4)
                    
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
                    
                    with col4:
                        st.metric("LLM Provider", selected_provider)
                        st.metric("Language", selected_language)

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                logger.error(f"Failed to process document '{uploaded_file.name}'.", exc_info=True)
                
                # Show troubleshooting tips
                with st.expander("🔧 Troubleshooting", expanded=True):
                    st.markdown(f"""
                    **Common issues with {selected_provider}:**
                    
                    - **API Key Missing**: Make sure you have set the `{provider_name.upper()}_API_KEY` environment variable
                    - **Rate Limits**: Try again in a few moments if you're hitting rate limits
                    - **Network Issues**: Check your internet connection
                    - **File Format**: Ensure your document is in a supported format
                    
                    **Environment Variables Required:**
                    - For Groq: `GROQ_API_KEY`
                    - For Gemini: `GEMINI_API_KEY`
                    """)
                
            finally:
                # Clean up temporary file
                if 'file_path' in locals() and file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Removed temporary file: {file_path}")
                
                # Restore original summarizer if it was changed
                if 'original_summarizer' in locals():
                    processor.summarizer = original_summarizer

    else:
        # Show supported file types when no file is uploaded
        st.info("📄 Please upload a document to begin processing.")

def render_batch_folder_processing(processor, base_prompt_dir, lang_code, provider_name, selected_provider, selected_language):
    """Render batch folder processing interface."""
    
    st.subheader("📁 Batch Process Folder")
    
    # Get document processing config
    try:
        doc_config = get_document_processing_config()
        default_knowledge_base = doc_config.get('knowledge_base_path', '/data/knowledge_base')
        supported_extensions = doc_config.get('supported_extensions', ['.pdf', '.md', '.markdown'])
    except Exception as e:
        logger.warning(f"Could not load document processing config: {e}")
        default_knowledge_base = '/data/knowledge_base'
        supported_extensions = ['.pdf', '.md', '.markdown']
    
    # Folder path input
    folder_path = st.text_input(
        "📂 Knowledge Base Folder Path",
        value=default_knowledge_base,
        help="Path to the folder containing documents to process"
    )
    
    # Check if folder exists and scan for documents
    if folder_path:
        if os.path.exists(folder_path):
            documents = find_documents_in_folder(folder_path, supported_extensions)
            
            if documents:
                st.success(f"✅ Found **{len(documents)}** supported documents in folder")
                
                # Show preview of documents
                with st.expander("📋 Documents to be processed", expanded=False):
                    for doc in documents[:10]:  # Show first 10
                        relative_path = doc.relative_to(Path(folder_path))
                        st.text(f"📄 {relative_path}")
                    
                    if len(documents) > 10:
                        st.text(f"... and {len(documents) - 10} more documents")
                
                # Processing options
                st.subheader("⚙️ Batch Processing Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    max_documents = st.number_input(
                        "Max documents to process",
                        min_value=1,
                        max_value=len(documents),
                        value=min(10, len(documents)),
                        help="Limit the number of documents to process"
                    )
                
                with col2:
                    skip_existing = st.checkbox(
                        "Skip existing documents",
                        value=True,
                        help="Skip documents that are already in the database"
                    )
                
                # Process button
                if st.button(f"🚀 Process {max_documents} Documents", use_container_width=True):
                    try:
                        # Create LLM provider
                        with st.spinner(f"🔄 Initializing {selected_provider}..."):
                            llm_provider = LLMProviderFactory.create_provider(provider_name)
                        
                        # Filter documents if needed
                        docs_to_process = documents[:max_documents]
                        
                        if skip_existing:
                            # Get existing document IDs from database
                            try:
                                existing_chunks = processor.sql_handler.get_all_chunks()
                                existing_doc_ids = set(chunk['document_id'] for chunk in existing_chunks)
                                docs_to_process = [doc for doc in docs_to_process 
                                                 if doc.name not in existing_doc_ids]
                                
                                if len(docs_to_process) < max_documents:
                                    skipped = max_documents - len(docs_to_process)
                                    st.info(f"ℹ️ Skipping {skipped} documents that are already processed")
                            except Exception as e:
                                logger.warning(f"Could not check existing documents: {e}")
                        
                        if docs_to_process:
                            st.info(f"🚀 Starting batch processing of {len(docs_to_process)} documents...")
                            process_multiple_documents(
                                docs_to_process, processor, llm_provider, 
                                base_prompt_dir, lang_code, selected_provider
                            )
                        else:
                            st.warning("⚠️ No new documents to process")
                            
                    except Exception as e:
                        st.error(f"❌ Failed to initialize processing: {str(e)}")
                        logger.error(f"Batch processing initialization failed: {e}")
            else:
                st.warning(f"⚠️ No supported documents found in folder")
                st.info(f"Supported extensions: {', '.join(supported_extensions)}")
        else:
            st.error(f"❌ Folder does not exist: `{folder_path}`")
    
    # Show help section
    with st.expander("ℹ️ Batch Processing Help", expanded=False):
        st.markdown(f"""
        **Batch Processing Features:**
        
        - **Recursive Search**: Scans all subfolders for supported documents
        - **Supported Formats**: {', '.join(supported_extensions)}
        - **Progress Tracking**: Real-time progress updates for each document
        - **Error Handling**: Failed documents are reported but don't stop the batch
        - **Skip Existing**: Option to skip documents already in the database
        
        **Tips:**
        - Start with a small batch (5-10 documents) to test your configuration
        - Monitor the progress and check for any failed documents
        - Use consistent naming for better organization
        - Ensure your API keys are properly configured before starting large batches
        """)

    # Show general help when no mode is active
    if not uploaded_file and processing_mode == "Single File Upload":
        show_general_help(supported_extensions)

def show_general_help(supported_extensions):
    """Show general help information."""
    st.info("📄 Please upload a document or switch to batch processing mode.")
    
    with st.expander("ℹ️ Supported File Types & Requirements", expanded=False):
        st.markdown(f"""
        **Supported document formats:**
        - **PDF files** (.pdf) - Extracts text, tables, and images
        - **Markdown files** (.md, .markdown) - Processes text, code blocks, tables, and image references
        
        **Processing capabilities:**
        - Text chunking with configurable size
        - Table extraction (as images for PDF, as markdown for MD)
        - Image processing and base64 conversion
        - Code block detection (Markdown only)
        - Multilingual summarization support
        
        **Requirements:**
        - **For Groq**: Set `GROQ_API_KEY` environment variable
        - **For Gemini**: Set `GEMINI_API_KEY` environment variable
        
        **Supported Languages:**
        - **English**: All features fully supported
        - **Español**: Complete Spanish summarization support
        """)
    
    # Show API key status
    with st.expander("🔑 API Key Status", expanded=False):
        import os
        
        col1, col2 = st.columns(2)
        
        with col1:
            groq_key = os.getenv('GROQ_API_KEY')
            if groq_key:
                st.success("✅ Groq API Key: Configured")
            else:
                st.error("❌ Groq API Key: Missing")
        
        with col2:
            gemini_key = os.getenv('GEMINI_API_KEY')
            if gemini_key:
                st.success("✅ Gemini API Key: Configured")
            else:
                st.error("❌ Gemini API Key: Missing")