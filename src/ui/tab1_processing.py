# src/ui/tab1_processing.py

import streamlit as st
from src.processing.document_processor import DocumentProcessor
from src.llm_provider.llm_factory import LLMProviderFactory
from src.processing.multimodal_summarizer import MultimodalSummarizer
from pathlib import Path
import tempfile
import logging

# Get a logger for the current module
logger = logging.getLogger(__name__)

def generate_unique_id() -> str:
    """Generate a unique ID for chunks."""
    import uuid
    return str(uuid.uuid4())

def render_tab1(processor: DocumentProcessor, base_prompt_dir: str):
    """
    Renders the 'Document Processing' tab in the Streamlit UI.

    This tab handles the document upload and the ingestion pipeline,
    providing granular visual feedback to the user throughout the process.
    Supports both PDF and Markdown files with configurable LLM provider and language.

    Args:
        processor (DocumentProcessor): An instance of the main DocumentProcessor.
        base_prompt_dir (str): Base directory for prompt templates.
    """
    logger.debug("Rendering Tab 1: Document Processing.")
    
    st.header("Step 1: Process a Document")
    st.markdown("""
    Upload a PDF or Markdown file. The system will parse it into text, table, and image chunks,
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
            "📝 Summary Language",
            options=list(language_options.keys()),
            index=0,
            help="Language for generating summaries"
        )
        lang_code = language_options[selected_language]
    
    with col2:
        # LLM Provider selector
        provider_options = {
            'Groq (Llama)': 'groq',
            'Google Gemini': 'gemini'
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
        st.info("📁 Please upload a document to begin processing.")
        
        with st.expander("ℹ️ Supported File Types & Requirements", expanded=False):
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