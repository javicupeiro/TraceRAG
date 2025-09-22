import streamlit as st
from core.embedder import Embedder
from database.sql_handler import SQLHandler
from database.vector_handler import VectorHandler
from llm_provider.llm_factory import LLMProviderFactory
from llm_provider.base_llm_provider import LLMChunk
from pathlib import Path
from typing import List, Tuple
import numpy as np
import time
import logging

# Get a logger for the current module
logger = logging.getLogger(__name__)

# --- Helper Functions for Metric Calculation ---
def cosine_similarity(v1, v2):
    """Calculates cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def l2_distance(v1, v2):
    """Calculates L2 (Euclidean) distance between two vectors."""
    return np.linalg.norm(np.array(v1) - np.array(v2))

def display_chunk(chunk: dict, key_prefix: str = "", search_metric: str = "cosine"):
    """
    Display a chunk with appropriate formatting based on its type.
    
    Args:
        chunk (dict): The chunk data from database
        key_prefix (str): Unique prefix for Streamlit keys
        search_metric (str): The search metric being used ('cosine' or 'l2')
    """
    chunk_type = chunk.get('chunk_type', 'unknown')
    chunk_id = chunk.get('id', 'unknown')
    
    with st.container():
        # Header with chunk info
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"**Chunk ID:** `{chunk_id}`")
        with col2:
            st.write(f"**Type:** {chunk_type.title()}")
        with col3:
            similarity_value = chunk.get('cosine_similarity', 0) if search_metric == 'cosine' else chunk.get('l2_distance', 0)
            metric_label = "Similarity" if search_metric == 'cosine' else "Distance"
            st.write(f"**{metric_label}:** {similarity_value:.3f}")
        
        # Content based on type
        if chunk_type == 'text':
            content = chunk.get('content', '')
            st.text_area(
                "Content:", 
                value=content[:500] + ("..." if len(content) > 500 else ""),
                height=100,
                key=f"{key_prefix}_content_{chunk_id}",
                disabled=True
            )
        
        elif chunk_type in ['image', 'table']:
            caption = chunk.get('chunk_metadata', {}).get('caption', 'No caption')
            st.write(f"**Caption:** {caption}")
            
            # Try to display image if it's base64
            try:
                import base64
                from PIL import Image
                import io
                
                image_data = chunk.get('content', '')
                if image_data:
                    # Remove data URL prefix if present
                    if image_data.startswith('data:'):
                        image_data = image_data.split(',')[1]
                    
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                    st.image(image, caption=caption, use_column_width=True)
            except Exception as e:
                st.write("🖼️ Image content (cannot display)")
                logger.debug(f"Could not display image: {e}")
        
        elif chunk_type == 'code':
            content = chunk.get('content', '')
            language = chunk.get('chunk_metadata', {}).get('language', '')
            st.code(content, language=language)
        
        # Summary
        summary = chunk.get('summary', '')
        if summary:
            st.write(f"**Summary:** {summary}")
        
        st.divider()

def build_multimodal_prompt_and_chunks(chunks: List[dict]) -> Tuple[str, List[LLMChunk]]:
    """
    Constructs a dynamic prompt and a list of LLMChunk objects for the multimodal LLM.
    
    Args:
        chunks (List[dict]): The list of retrieved chunks to be used as context.

    Returns:
        A tuple containing:
        - str: A formatted string containing all text-based context.
        - List[LLMChunk]: A list of LLMChunk objects to be sent to the LLM.
    """
    prompt_context_parts = []
    llm_chunks = []

    for chunk in chunks:
        chunk_id = chunk.get('id', 'unknown')
        chunk_type = chunk.get('chunk_type', 'text')
        content = chunk.get('content', '')
        metadata = chunk.get('chunk_metadata', {})
        
        # Enhanced: Add document information
        document_id = chunk.get('document_id', 'Unknown Document')
        source_page = chunk.get('source_page', 'N/A')
        
        if chunk_type == 'text':
            # Include document info in the context
            document_info = f"[From: {document_id}, Page: {source_page}]" if source_page != 'N/A' else f"[From: {document_id}]"
            prompt_context_parts.append(f"--- Text Context {document_info} ---\n{content}")
            llm_chunks.append(LLMChunk(content=content, type="text", metadata=metadata))
        
        elif chunk_type == 'table':
            caption = metadata.get('caption', 'No caption provided').strip()
            if not caption:
                caption = "Untitled table"
            
            document_info = f"[From: {document_id}, Page: {source_page}]" if source_page != 'N/A' else f"[From: {document_id}]"
            prompt_context_parts.append(
                f"--- Table Context {document_info} ---\n"
                f"A table titled \"{caption}\" is provided below."
            )
            
            # Determine if it's base64 image or markdown table
            try:
                import base64
                base64.b64decode(content)
                llm_chunks.append(LLMChunk(content=content, type="image", metadata=metadata))
            except:
                llm_chunks.append(LLMChunk(content=content, type="table", metadata=metadata))
        
        elif chunk_type == 'image':
            caption = metadata.get('caption', 'No caption provided').strip()
            if not caption:
                caption = "Untitled image"
            
            document_info = f"[From: {document_id}, Page: {source_page}]" if source_page != 'N/A' else f"[From: {document_id}]"
            prompt_context_parts.append(
                f"--- Image Context {document_info} ---\n"
                f"An image titled \"{caption}\" is provided for visual analysis."
            )
            llm_chunks.append(LLMChunk(content=content, type="image", metadata=metadata))

    final_context_str = "\n\n".join(prompt_context_parts)
    return final_context_str, llm_chunks
    

def render_tab3(embedder: Embedder, 
                vector_handler: VectorHandler, 
                sql_handler: SQLHandler, 
                base_prompt_dir: str):
    """
    Renders the 'Chat with Documents' tab in the Streamlit UI.

    This tab contains the main RAG chat interface, handling user queries,
    retrieval, context assembly, and response generation.

    Args:
        embedder: The embedding model instance.
        vector_handler: The handler for the vector database.
        sql_handler: The handler for the SQL database.
        base_prompt_dir (str): The base directory for prompt templates.
    """
    logger.debug("Rendering Tab 3: Chat with Documents.")
    
    st.header("💬 Chat with your documents")
    st.markdown("Ask a question, and the system will retrieve relevant text and images to form an answer.")

    # Configuration Section
    st.subheader("🔧 Chat Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Language selector
        language_options = {
            'English': 'en',
            'Español': 'es'
        }
        selected_language = st.selectbox(
            "🌍 Response Language",
            options=list(language_options.keys()),
            index=0,
            help="Language for chat responses"
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
            help="Choose the language model provider for responses"
        )
        provider_name = provider_options[selected_provider]
    
    with col3:
        # Search metric selector
        metric_options = {
            'Cosine Similarity': 'cosine',
            'Euclidean Distance': 'l2'
        }
        selected_metric = st.selectbox(
            "🔍 Search Metric",
            options=list(metric_options.keys()),
            index=0,
            help="Similarity metric for ranking retrieved documents"
        )
        search_metric = metric_options[selected_metric]

    # Load the correct chat prompt based on the selected language
    prompt_template_path = Path(base_prompt_dir) / lang_code / "query_context.txt"
    try:
        with open(prompt_template_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
    except FileNotFoundError:
        st.error(f"Chat prompt file not found at: {prompt_template_path}")
        logger.error(f"Could not load chat prompt from {prompt_template_path}")
        return

    st.divider()

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages from history
    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 View used sources"):
                    for src_idx, chunk in enumerate(message["sources"]):
                        display_chunk(chunk, key_prefix=f"history_{msg_idx}_{src_idx}")

    # Handle new user input
    if query := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        logger.info(f"User asked a new query: '{query}' using {provider_name}")

        # Initialize original_chunks variable BEFORE the chat_message block
        original_chunks = []
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching and thinking..."):
                try:
                    # --- RAG Pipeline Execution ---
                    # 1. Embed the user's query
                    query_embedding = embedder.embed([query])[0]
                    
                    # 2. Perform vector search (always use IP for Milvus, we'll re-rank locally)
                    search_results = vector_handler.search_similar(query_embedding, k=10, metric_type="IP")
                    
                    if not search_results:
                        st.warning("Could not find any relevant chunks in the documents.")
                        logger.warning(f"No relevant chunks found for query: '{query}'")
                        return
                    
                    logger.info(f"Retrieved {len(search_results)} candidate chunks.")
                    
                    # 3. Retrieve full data and calculate both metrics locally
                    retrieved_ids = [res['id'] for res in search_results]
                    vector_map = vector_handler.get_vectors_by_ids(retrieved_ids)
                    original_chunks = sql_handler.get_chunks_by_ids(retrieved_ids)

                    # Calculate both similarity metrics for each chunk
                    for chunk in original_chunks:
                        chunk_vector = vector_map.get(chunk['id'])
                        if chunk_vector:
                            chunk['cosine_similarity'] = cosine_similarity(query_embedding, chunk_vector)
                            chunk['l2_distance'] = l2_distance(query_embedding, chunk_vector)
                        else:
                            chunk['cosine_similarity'] = 0.0
                            chunk['l2_distance'] = float('inf')
                    
                    # 4. Sort by the user-selected metric
                    if search_metric == 'cosine':
                        original_chunks.sort(key=lambda x: x['cosine_similarity'], reverse=True)
                        # Take top 5 after re-ranking
                        original_chunks = original_chunks[:5]
                    else:  # l2
                        original_chunks.sort(key=lambda x: x['l2_distance'])
                        # Take top 5 after re-ranking
                        original_chunks = original_chunks[:5]
                    
                    # 4. Create LLM provider
                    with st.spinner(f"🤖 Initializing {selected_provider}..."):
                        try:
                            llm_provider = LLMProviderFactory.create_provider(provider_name)
                            logger.info(f"Successfully created {provider_name} provider")
                        except Exception as e:
                            st.error(f"❌ Failed to initialize {selected_provider}: {str(e)}")
                            st.error("Please check your API keys in environment variables")
                            logger.error(f"Failed to create {provider_name} provider: {e}")
                            return
                    
                    # 5. Build multimodal prompt and chunks
                    context_str, llm_chunks = build_multimodal_prompt_and_chunks(original_chunks)
                    formatted_prompt = prompt_template.format(context=context_str, query=query)
                    
                    # 6. Generate response from LLM
                    with st.spinner(f"💭 Generating response with {selected_provider}..."):
                        response = llm_provider.answer_query(llm_chunks, formatted_prompt)

                    # 7. Display response and sources
                    st.markdown(response.content)
                    
                    # Show response metadata
                    with st.expander("ℹ️ Response Info", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Model Used", response.model_used)
                        with col2:
                            if response.tokens_used:
                                st.metric("Tokens Used", response.tokens_used)
                        with col3:
                            if response.response_time:
                                st.metric("Response Time", f"{response.response_time:.2f}s")
                    
                    # 8. Show sources - NOW original_chunks is defined
                    if original_chunks:
                        with st.expander("📚 View used sources"):
                            for src_idx, chunk in enumerate(original_chunks):
                                display_chunk_with_document_info(chunk, key_prefix=f"current_{src_idx}", search_metric=search_metric)
                    
                    # 9. Add response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response.content, 
                        "sources": original_chunks,
                        "metadata": {
                            "provider": selected_provider,
                            "language": selected_language,
                            "model": response.model_used,
                            "tokens": response.tokens_used,
                            "response_time": response.response_time
                        }
                    })
                    
                except Exception as e:
                    st.error(f"An error occurred during chat: {e}")
                    logger.error(f"Chat error: {e}", exc_info=True)

    # Show help when no messages
    if not st.session_state.messages:
        st.info("👋 Start a conversation by asking a question about your documents!")

def display_chunk_with_document_info(chunk: dict, key_prefix: str = "", search_metric: str = "cosine"):
    """Display a chunk with enhanced document information."""
    
    chunk_type = chunk.get('chunk_type', 'unknown')
    chunk_id = chunk.get('id', 'unknown')
    document_id = chunk.get('document_id', 'Unknown Document')
    source_page = chunk.get('source_page', 'N/A')
    
    with st.container():
        # Enhanced header with document info
        st.markdown(f"### 📄 {document_id}")
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            page_info = f"Page {source_page}" if source_page != 'N/A' else "Main content"
            st.write(f"**Section:** {page_info}")
        with col2:
            st.write(f"**Type:** {chunk_type.title()}")
        with col3:
            if search_metric == 'cosine':
                similarity_value = chunk.get('cosine_similarity', 0)
                st.write(f"**Similarity:** {similarity_value:.3f}")
            else:
                distance_value = chunk.get('l2_distance', 0)
                st.write(f"**Distance:** {distance_value:.3f}")
        with col4:
            st.write(f"**ID:** `{chunk_id[:8]}...`")
        
        # Content based on type
        if chunk_type == 'text':
            content = chunk.get('content', '')
            st.text_area(
                "Content:", 
                value=content[:500] + ("..." if len(content) > 500 else ""),
                height=100,
                key=f"{key_prefix}_content_{chunk_id}",
                disabled=True
            )
        
        elif chunk_type in ['image', 'table']:
            caption = chunk.get('chunk_metadata', {}).get('caption', 'No caption')
            st.write(f"**Caption:** {caption}")
            
            # Try to display image if it's base64
            try:
                import base64
                from PIL import Image
                import io
                
                image_data = chunk.get('content', '')
                if image_data:
                    # Remove data URL prefix if present
                    if image_data.startswith('data:'):
                        image_data = image_data.split(',')[1]
                    
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                    st.image(image, caption=caption, use_column_width=True)
            except Exception as e:
                st.write("🖼️ Image content (cannot display)")
                logger.debug(f"Could not display image: {e}")
        
        # Summary
        summary = chunk.get('summary', '')
        if summary:
            st.write(f"**Summary:** {summary}")
        
        st.divider()
