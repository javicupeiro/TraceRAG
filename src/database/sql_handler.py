import logging
from pathlib import Path
from typing import List, Dict, Any

from sqlalchemy import create_engine, Column, String, Integer, Text, JSON
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# Get a logger for the current module
logger = logging.getLogger(__name__)

# --- SQLAlchemy Model Definition ---
class Base(DeclarativeBase):
    """Base class for SQLAlchemy declarative models."""
    pass

class DocumentChunkModel(Base):
    """
    SQLAlchemy model representing a single document chunk in the database.
    
    This table stores the original content (text or base64 image) and its metadata,
    linking it to a vector in Milvus via its unique 'id'.
    """
    __tablename__ = 'document_chunks'

    id = Column(String, primary_key=True, index=True, comment="Unique ID, links to Milvus vector.")
    document_id = Column(String, nullable=False, index=True, comment="Identifier of the source document.")
    source_page = Column(Integer, nullable=False, comment="Page number in the source document.")
    chunk_type = Column(String, nullable=False, comment="Type of chunk: 'text', 'table', or 'image'.")
    content = Column(Text, nullable=False, comment="Original content: text, or base64 for images/tables.")
    summary = Column(Text, nullable=True, comment="LLM-generated summary of the content.")
    chunk_metadata = Column(JSON, comment="Additional metadata, e.g., captions.")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the SQLAlchemy object to a Python dictionary."""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

# --- SQL Handler Class ---
class SQLHandler:
    """
    Manages all interactions with the SQLite database for storing and retrieving
    original document chunks.
    """
    def __init__(self, db_path: str | Path):
        """
        Initializes the database connection and creates the table if it doesn't exist.

        Args:
            db_path (str | Path): The file path for the SQLite database.
        """
        logger.debug(f"Initializing SQLHandler for database at: {db_path}")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 'check_same_thread' is required for SQLite with multi-threaded apps like Streamlit.
        db_url = f'sqlite:///{self.db_path}'
        self.engine = create_engine(db_url, connect_args={"check_same_thread": False})
        self.Session = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        self._create_table()
        logger.info(f"SQLHandler initialized successfully for database: {db_url}")

    def _create_table(self):
        """Creates the 'document_chunks' table if it does not already exist."""
        try:
            logger.debug("Verifying 'document_chunks' table existence.")
            Base.metadata.create_all(bind=self.engine)
            logger.info("Table 'document_chunks' is ready.")
        except Exception as e:
            logger.error(f"Error creating or verifying database table: {e}")
            raise

    def add_chunk(self, chunk_id: str, document_id: str, chunk_type: str, 
                  content: str, source_page: int, chunk_metadata: Dict[str, Any], summary: str):
        """
        Adds a new document chunk record to the database.

        Args:
            chunk_id (str): The unique ID for the chunk.
            document_id (str): The identifier for the source document (e.g., filename).
            chunk_type (str): The type of the chunk ('text', 'table', 'image').
            content (str): The actual content of the chunk.
            source_page (int): The page number from which the chunk was extracted.
            chunk_metadata (Dict[str, Any]): A dictionary for any additional metadata.
            summary (str): The LLM-generated summary of the chunk.
        """
        logger.debug(f"Adding chunk with ID: {chunk_id} to the database.")
        new_chunk = DocumentChunkModel(
            id=chunk_id, document_id=document_id, chunk_type=chunk_type,
            content=content, source_page=source_page,
            chunk_metadata=chunk_metadata, summary=summary
        )
        try:
            with self.Session() as session:
                session.add(new_chunk)
                session.commit()
            logger.info(f"Successfully added chunk with ID: {chunk_id}")
        except Exception as e:
            logger.error(f"Failed to add chunk with ID {chunk_id}: {e}")
            raise

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieves multiple chunks from the database based on a list of IDs.
        
        Args:
            chunk_ids (List[str]): A list of unique chunk IDs to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of chunks, where each chunk is a dictionary.
        """
        if not chunk_ids:
            return []
        logger.info(f"Retrieving {len(chunk_ids)} chunk(s) by ID from the database.")
        try:
            with self.Session() as session:
                chunks = session.query(DocumentChunkModel).filter(DocumentChunkModel.id.in_(chunk_ids)).all()
                logger.debug(f"Found {len(chunks)} matching chunks.")
                return [chunk.to_dict() for chunk in chunks]
        except Exception as e:
            logger.error(f"Failed to retrieve chunks by IDs: {e}")
            raise

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Retrieves all chunk records from the database."""
        logger.info("Retrieving all chunks from the database.")
        try:
            with self.Session() as session:
                chunks = session.query(DocumentChunkModel).all()
                logger.info(f"Found a total of {len(chunks)} chunks.")
                return [chunk.to_dict() for chunk in chunks]
        except Exception as e:
            logger.error(f"Failed to retrieve all chunks: {e}")
            raise

    def delete_all_data(self):
        """Deletes all records from the 'document_chunks' table."""
        logger.warning("Attempting to delete all records from the 'document_chunks' table.")
        try:
            with self.Session() as session:
                num_rows_deleted = session.query(DocumentChunkModel).delete()
                session.commit()
                logger.info(f"Successfully deleted {num_rows_deleted} records from the database.")
        except Exception as e:
            logger.error(f"Failed to delete all data from SQL database: {e}")
            raise

    def get_stats(self) -> Dict[str, int]:
        """
        Calculates statistics about the database content.

        Returns:
            Dict[str, int]: A dictionary with the total number of chunks.
        """
        logger.debug("Getting database statistics.")
        try:
            with self.Session() as session:
                total_chunks = session.query(DocumentChunkModel).count()
            return {"total_chunks": total_chunks}
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"total_chunks": -1}