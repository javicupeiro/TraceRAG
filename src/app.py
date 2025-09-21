import logging
import streamlit as st
import sys
from pathlib import Path

# --- Basic Logging Configuration ---
# To see DEBUG messages, you might need to adjust the level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- System Path Configuration ---
# This ensures that the application can be run from the root directory
FILE = Path(__file__).resolve()
ROOT = FILE.parent  # Project root is where app.py is located
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
logger.debug(f"Project root added to system path: {ROOT}")

# --- Backend Component Imports ---
from src.core.embedder import Embedder
from src.core.config_manager import get_config_manager
from src.database.sql_handler import SQLHandler
from src.database.vector_handler import VectorHandler
from src.processing.document_processor import DocumentProcessor
from src.llm_provider.llm_factory import LLMProviderFactory
from src.processing.multimodal_summarizer import MultimodalSummarizer

# --- UI Tab Module Imports ---
from src.ui.tab1_processing import render_tab1
from src.ui.tab3_chat import render_tab3

# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent Support System",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def initialize_services() -> tuple:
    """
    Initializes and caches all backend services for the application.
    This function is decorated with @st.cache_resource to ensure that
    heavy objects like models and database connections are created only once.

    Returns:
        tuple: A tuple containing all initialized service handlers.
    """
    logger.info("Initializing backend services...")
    
    # Get configuration
    config_manager = get_config_manager()
    
    # Database Handlers
    db_config = config_manager.get_database_config()
    sql_handler = SQLHandler(db_path=db_config['sql_path'])
    vector_handler = VectorHandler(
        host=db_config['milvus_host'],
        port=db_config['milvus_port'],
        collection_name=db_config['collection_name'],
        vector_dim=config_manager.get_embedding_config()['dimensions']
    )
    
    # Core AI Models
    embedding_config = config_manager.get_embedding_config()
    embedder = Embedder(model_name=embedding_config['name'])
    
    # Create a default LLM provider for DocumentProcessor initialization
    # This will be overridden in the UI when users select their preferred provider
    try:
        default_llm_provider = LLMProviderFactory.create_provider('groq')
        logger.info("Successfully created default Groq provider")
    except Exception as e:
        logger.warning(f"Failed to create default Groq provider: {e}")
        try:
            default_llm_provider = LLMProviderFactory.create_provider('gemini')
            logger.info("Successfully created default Gemini provider")
        except Exception as e2:
            logger.error(f"Failed to create any default LLM provider: {e2}")
            raise Exception("No LLM provider available. Please check your API keys.")
    
    # Document Processing Logic
    base_prompt_dir = "prompts"  # Default prompt directory
    summarizer = MultimodalSummarizer(
        llm_provider=default_llm_provider, 
        base_prompt_dir=base_prompt_dir
    )
    
    processor = DocumentProcessor(
        summarizer=summarizer,
        embedder=embedder,
        sql_handler=sql_handler,
        vector_handler=vector_handler
    )
    
    logger.info("All backend services initialized successfully.")
    return sql_handler, vector_handler, embedder, processor, base_prompt_dir

def main():
    """Main function to run the Streamlit application."""
    st.title("🤖 Intelligent Technical Support System")
    st.caption("Process documents and chat with your knowledge base using advanced AI models.")

    try:
        sql_handler, vector_handler, embedder, processor, base_prompt_dir = initialize_services()
    except Exception as e:
        st.error(f"Fatal error during application initialization: {e}")
        logger.exception("Application initialization failed.")
        st.error("**Please check:**")
        st.error("- Your API keys are set in environment variables (GROQ_API_KEY or GEMINI_API_KEY)")
        st.error("- Milvus database is running (docker-compose up)")
        st.error("- All required dependencies are installed")
        st.stop()
    
    # --- Sidebar for Database Management ---
    st.sidebar.header("📊 Database Management")
    
    # Show database statistics
    try:
        sql_stats = sql_handler.get_stats()
        vector_stats = vector_handler.get_stats()
        
        st.sidebar.metric("Total Chunks", sql_stats.get('total_chunks', 0))
        st.sidebar.metric("Total Vectors", vector_stats.get('total_vectors', 0))
    except Exception as e:
        st.sidebar.error("Could not load database stats")
        logger.error(f"Failed to load database stats: {e}")
    
    st.sidebar.divider()
    
    # Clear database functionality
    st.sidebar.warning("⚠️ Danger Zone")
    
    if st.sidebar.button("🗑️ Clear ALL databases", use_container_width=True):
        @st.dialog("Confirmation Required")
        def confirm_delete():
            st.write("⚠️ **This will permanently delete all processed documents and their embeddings.**")
            st.write("Are you absolutely sure you want to continue?")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("❌ Cancel", use_container_width=True):
                    st.rerun()
            with col2:
                if st.button("✅ Yes, delete everything", type="primary", use_container_width=True):
                    with st.spinner("Clearing databases..."):
                        try:
                            sql_handler.delete_all_data()
                            vector_handler.delete_all_data()
                            st.session_state.confirm_delete = True
                            logger.info("All databases cleared by user")
                        except Exception as e:
                            st.error(f"Error clearing databases: {e}")
                            logger.error(f"Failed to clear databases: {e}")
                    st.rerun()
        
        confirm_delete()
        
        if st.session_state.get("confirm_delete"):
            st.sidebar.success("✅ All databases have been cleared.")
            del st.session_state["confirm_delete"]
            st.rerun()
 
    # --- API Keys Status ---
    st.sidebar.divider()
    st.sidebar.header("🔑 API Keys Status")
    
    import os
    
    # Check Groq API key
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key:
        st.sidebar.success("✅ Groq API Key")
    else:
        st.sidebar.error("❌ Groq API Key")
    
    # Check Gemini API key
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key:
        st.sidebar.success("✅ Gemini API Key")
    else:
        st.sidebar.error("❌ Gemini API Key")
    
    if not groq_key and not gemini_key:
        st.sidebar.error("⚠️ No API keys found!")
        st.sidebar.info("Set GROQ_API_KEY or GEMINI_API_KEY environment variables")
    
    # --- Main Application Tabs ---
    tab1, tab2 = st.tabs([
        "📄 Document Processing",
        "💬 Chat with Documents"
    ])

    with tab1:
        render_tab1(processor, base_prompt_dir)
    
    with tab2:
        render_tab3(embedder, vector_handler, sql_handler, base_prompt_dir)

if __name__ == "__main__":
    logger.info("Starting Intelligent Support System application...")
    main()