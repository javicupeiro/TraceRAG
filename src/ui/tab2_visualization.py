import streamlit as st
import pandas as pd
from database.sql_handler import SQLHandler
from database.vector_handler import VectorHandler
import logging

# Get a logger for the current module
logger = logging.getLogger(__name__)

def display_chunk(chunk: dict, key_prefix: str = ""):
    """
    Display a chunk with appropriate formatting based on its type.
    
    Args:
        chunk (dict): The chunk data from database
        key_prefix (str): Unique prefix for Streamlit keys
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
            source_page = chunk.get('source_page', 'N/A')
            st.write(f"**Page:** {source_page}")
        
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
        
        # Document metadata
        document_id = chunk.get('document_id', 'Unknown')
        st.write(f"**Document:** {document_id}")
        
        st.divider()

def render_tab2(sql_handler: SQLHandler, vector_handler: VectorHandler):
    """
    Renders the 'Chunks Visualization' tab in the Streamlit UI.

    This tab displays database statistics and allows users to browse and inspect
    the individual chunks that have been processed and stored.

    Args:
        sql_handler (SQLHandler): The handler for the SQL database.
        vector_handler (VectorHandler): The handler for the vector database.
    """
    logger.debug("Rendering Tab 2: Chunks Visualization.")
    
    st.header("📊 Explore Processed Data")
    st.markdown("View database statistics and explore stored chunks by clicking on them.")

    # Display database statistics
    try:
        col1, col2 = st.columns(2)
        with col1:
            sql_stats = sql_handler.get_stats()
            st.metric("Chunks in SQL DB", sql_stats.get('total_chunks', 0))
        with col2:
            vector_stats = vector_handler.get_stats()
            st.metric("Vectors in Milvus DB", vector_stats.get('total_vectors', 0))
    except Exception as e:
        st.error("Could not retrieve database statistics.")
        logger.error(f"Failed to get database stats: {e}", exc_info=True)

    st.subheader("📄 Stored Chunks")
    
    try:
        all_chunks = sql_handler.get_all_chunks()
        if not all_chunks:
            st.warning("No chunks found in the database. Please process a document in the Document Processing tab.")
            return

        # Use pandas for a clean, searchable table display
        df = pd.DataFrame(all_chunks)
        df_display = df[['id', 'document_id', 'chunk_type', 'source_page', 'summary']]
                
        st.info("Click on a row in the table below to see the full chunk details.")
        
        # The interactive dataframe for chunk selection
        selection = st.dataframe(
            df_display, on_select="rerun", selection_mode="single-row",
            hide_index=True, width="stretch", key="chunk_selector"
        )
        
        # Display details of the selected chunk
        if selection.selection.rows:
            selected_row_index = selection.selection.rows[0]
            selected_chunk = all_chunks[selected_row_index]
            logger.debug(f"User selected chunk with ID: {selected_chunk.get('id')}")
            
            st.divider()
            # A unique key_prefix is needed to avoid widget ID collisions
            display_chunk(selected_chunk, key_prefix="visualization_view")

    except Exception as e:
        st.error("Failed to load and display chunks from the database.")
        logger.error(f"An error occurred while rendering the chunks table: {e}", exc_info=True)