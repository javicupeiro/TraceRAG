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

class UserModel(Base):
    """
    SQLAlchemy model for system users.
    """
    __tablename__ = 'users'

    user_id = Column(String, primary_key=True, index=True, comment="Unique user identifier")
    name = Column(String, nullable=False, comment="User display name")
    role = Column(String, nullable=False, comment="User role (CEO, Marketing Director, etc.)")
    company_type = Column(String, nullable=False, comment="Company type (startup, enterprise, etc.)")
    language_preference = Column(String, nullable=False, default='en', comment="Preferred language (en/es)")
    created_at = Column(String, nullable=False, comment="User creation timestamp")
    last_active = Column(String, nullable=True, comment="Last activity timestamp")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the SQLAlchemy object to a Python dictionary."""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class QueryHistoryModel(Base):
    """
    SQLAlchemy model for user query history.
    """
    __tablename__ = 'query_history'

    query_id = Column(String, primary_key=True, index=True, comment="Unique query identifier")
    user_id = Column(String, nullable=False, index=True, comment="User who made the query")
    query_text = Column(Text, nullable=False, comment="The actual query text")
    category = Column(String, nullable=True, comment="Query category (hiring_process, ai_features, etc.)")
    intent = Column(String, nullable=True, comment="Query intent (information_seeking, decision_support, etc.)")
    timestamp = Column(String, nullable=False, comment="When the query was made")
    session_id = Column(String, nullable=True, comment="Session identifier for grouping queries")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the SQLAlchemy object to a Python dictionary."""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class RecommendationModel(Base):
    """
    SQLAlchemy model for generated recommendations.
    """
    __tablename__ = 'recommendations'

    recommendation_id = Column(String, primary_key=True, index=True, comment="Unique recommendation identifier")
    user_id = Column(String, nullable=False, index=True, comment="User who received the recommendation")
    query_id = Column(String, nullable=True, comment="Query that triggered this recommendation")
    resource_id = Column(String, nullable=False, comment="ID of recommended resource")
    rank = Column(Integer, nullable=False, comment="Rank in recommendation list (1, 2, 3)")
    relevance_score = Column(String, nullable=False, comment="Relevance score as JSON string")
    final_score = Column(String, nullable=False, comment="Final recommendation score")
    primary_reason = Column(Text, nullable=False, comment="Primary reason for recommendation")
    detailed_explanation = Column(Text, nullable=False, comment="Detailed explanation")
    timestamp = Column(String, nullable=False, comment="When recommendation was generated")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the SQLAlchemy object to a Python dictionary."""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class RecommendationFeedbackModel(Base):
    """
    SQLAlchemy model for recommendation feedback.
    """
    __tablename__ = 'recommendation_feedback'

    feedback_id = Column(String, primary_key=True, index=True, comment="Unique feedback identifier")
    recommendation_id = Column(String, nullable=False, index=True, comment="Recommendation being rated")
    user_id = Column(String, nullable=False, comment="User providing feedback")
    feedback_type = Column(String, nullable=False, comment="Type of feedback (helpful, not_helpful, clicked)")
    timestamp = Column(String, nullable=False, comment="When feedback was given")
    additional_notes = Column(Text, nullable=True, comment="Optional additional feedback notes")

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
        """Creates all database tables if they do not already exist."""
        try:
            logger.debug("Verifying database tables existence.")
            Base.metadata.create_all(bind=self.engine)
            logger.info("All database tables are ready.")
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

    def add_user(self, user_id: str, name: str, role: str, company_type: str, language: str = 'en') -> bool:
        """
        Add a new user to the system.

        Args:
            user_id (str): Unique user identifier
            name (str): User display name
            role (str): User role (CEO, Marketing Director, etc.)
            company_type (str): Company type (startup, enterprise, etc.)
            language (str): Preferred language (en/es)

        Returns:
            bool: True if user was added successfully, False otherwise
        """
        logger.debug(f"Adding new user: {user_id} ({name})")

        from datetime import datetime
        current_time = datetime.now().isoformat()

        new_user = UserModel(
            user_id=user_id,
            name=name,
            role=role,
            company_type=company_type,
            language_preference=language,
            created_at=current_time,
            last_active=current_time
        )

        try:
            with self.Session() as session:
                # Check if user already exists
                existing_user = session.query(UserModel).filter(UserModel.user_id == user_id).first()
                if existing_user:
                    logger.warning(f"User {user_id} already exists")
                    return False

                session.add(new_user)
                session.commit()
                logger.info(f"Successfully added user: {user_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to add user {user_id}: {e}")
            return False

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve user information by user_id.

        Args:
            user_id (str): Unique user identifier

        Returns:
            Optional[Dict[str, Any]]: User data dictionary or None if not found
        """
        logger.debug(f"Retrieving user: {user_id}")

        try:
            with self.Session() as session:
                user = session.query(UserModel).filter(UserModel.user_id == user_id).first()
                if user:
                    logger.debug(f"Found user: {user_id}")
                    return user.to_dict()
                else:
                    logger.debug(f"User not found: {user_id}")
                    return None
        except Exception as e:
            logger.error(f"Failed to retrieve user {user_id}: {e}")
            return None

    def add_query_to_history(self, user_id: str, query_text: str, category: str = None, 
                            intent: str = None, session_id: str = None) -> str:
        """
        Add a user query to the history.

        Args:
            user_id (str): User who made the query
            query_text (str): The actual query text
            category (str): Query category (optional)
            intent (str): Query intent (optional)
            session_id (str): Session identifier (optional)

        Returns:
            str: Query ID if successful, empty string if failed
        """
        import uuid
        from datetime import datetime

        query_id = str(uuid.uuid4())
        current_time = datetime.now().isoformat()

        logger.debug(f"Adding query to history for user {user_id}: '{query_text[:50]}...'")

        new_query = QueryHistoryModel(
            query_id=query_id,
            user_id=user_id,
            query_text=query_text,
            category=category,
            intent=intent,
            timestamp=current_time,
            session_id=session_id
        )

        try:
            with self.Session() as session:
                session.add(new_query)
                session.commit()

                # Update user's last_active
                user = session.query(UserModel).filter(UserModel.user_id == user_id).first()
                if user:
                    user.last_active = current_time
                    session.commit()

                logger.info(f"Successfully added query {query_id} for user {user_id}")
                return query_id
        except Exception as e:
            logger.error(f"Failed to add query for user {user_id}: {e}")
            return ""

    def get_user_query_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get user's query history, ordered by most recent first.

        Args:
            user_id (str): User identifier
            limit (int): Maximum number of queries to return (default 10)

        Returns:
            List[Dict[str, Any]]: List of query dictionaries, empty list if none found
        """
        logger.debug(f"Retrieving query history for user {user_id} (limit: {limit})")

        try:
            with self.Session() as session:
                queries = session.query(QueryHistoryModel)\
                               .filter(QueryHistoryModel.user_id == user_id)\
                               .order_by(QueryHistoryModel.timestamp.desc())\
                               .limit(limit)\
                               .all()

                result = [query.to_dict() for query in queries]
                logger.debug(f"Retrieved {len(result)} queries for user {user_id}")
                return result
        except Exception as e:
            logger.error(f"Failed to retrieve query history for user {user_id}: {e}")
            return []

    def save_recommendations(self, user_id: str, query_id: str, recommendations: List[Any]) -> bool:
        """
        Save generated recommendations to database.

        Args:
            user_id (str): User who received recommendations
            query_id (str): Query that triggered recommendations
            recommendations (List[Any]): List of Recommendation objects

        Returns:
            bool: True if saved successfully, False otherwise
        """
        logger.debug(f"Saving {len(recommendations)} recommendations for user {user_id}")

        import uuid
        from datetime import datetime
        current_time = datetime.now().isoformat()

        try:
            with self.Session() as session:
                for rec in recommendations:
                    recommendation_id = str(uuid.uuid4())

                    new_recommendation = RecommendationModel(
                        recommendation_id=recommendation_id,
                        user_id=user_id,
                        query_id=query_id,
                        resource_id=rec.resource.resource_id,
                        rank=rec.rank,
                        relevance_score=str(rec.relevance_score),  # Store as string for simplicity
                        final_score=str(rec.final_score),
                        primary_reason=rec.primary_reason,
                        detailed_explanation=rec.detailed_explanation,
                        timestamp=current_time
                    )
                    session.add(new_recommendation)

                session.commit()
                logger.info(f"Successfully saved {len(recommendations)} recommendations")
                return True
        except Exception as e:
            logger.error(f"Failed to save recommendations for user {user_id}: {e}")
            return False

    def save_feedback(self, recommendation_id: str, user_id: str, feedback_type: str, 
                     additional_notes: str = None) -> bool:
        """
        Save user feedback for a recommendation.

        Args:
            recommendation_id (str): ID of the recommendation being rated
            user_id (str): User providing the feedback
            feedback_type (str): Type of feedback ('helpful', 'not_helpful', 'clicked')
            additional_notes (str): Optional additional feedback text

        Returns:
            bool: True if feedback saved successfully, False otherwise
        """
        logger.debug(f"Saving feedback for recommendation {recommendation_id}: {feedback_type}")

        import uuid
        from datetime import datetime

        feedback_id = str(uuid.uuid4())
        current_time = datetime.now().isoformat()

        new_feedback = RecommendationFeedbackModel(
            feedback_id=feedback_id,
            recommendation_id=recommendation_id,
            user_id=user_id,
            feedback_type=feedback_type,
            timestamp=current_time,
            additional_notes=additional_notes
        )

        try:
            with self.Session() as session:
                session.add(new_feedback)
                session.commit()
                logger.info(f"Successfully saved feedback {feedback_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to save feedback for recommendation {recommendation_id}: {e}")
            return False

    def get_recommendation_metrics(self) -> Dict[str, Any]:
        """
        Get basic metrics about the recommendation system performance.

        Returns:
            Dict[str, Any]: Dictionary with recommendation system metrics
        """
        logger.debug("Getting recommendation system metrics")

        try:
            with self.Session() as session:
                # Total recommendations generated
                total_recommendations = session.query(RecommendationModel).count()

                # Total feedback received
                total_feedback = session.query(RecommendationFeedbackModel).count()

                # Positive feedback count
                positive_feedback = session.query(RecommendationFeedbackModel)\
                                        .filter(RecommendationFeedbackModel.feedback_type == 'helpful')\
                                        .count()

                # Active users (users with queries)
                active_users = session.query(QueryHistoryModel.user_id).distinct().count()

                # Total queries
                total_queries = session.query(QueryHistoryModel).count()

                # Calculate satisfaction rate
                satisfaction_rate = 0.0
                if total_feedback > 0:
                    satisfaction_rate = positive_feedback / total_feedback

                metrics = {
                    "total_recommendations": total_recommendations,
                    "total_feedback": total_feedback,
                    "positive_feedback": positive_feedback,
                    "satisfaction_rate": satisfaction_rate,
                    "active_users": active_users,
                    "total_queries": total_queries,
                    "avg_queries_per_user": total_queries / active_users if active_users > 0 else 0
                }

                logger.info(f"Retrieved recommendation metrics: {metrics}")
                return metrics
        except Exception as e:
            logger.error(f"Failed to get recommendation metrics: {e}")
            return {
                "total_recommendations": -1,
                "total_feedback": -1,
                "positive_feedback": -1,
                "satisfaction_rate": 0.0,
                "active_users": -1,
                "total_queries": -1,
                "avg_queries_per_user": 0.0
            }

    def delete_all_recommendation_data(self):
        """
        Delete all recommendation-related data from database.

        WARNING: This will delete all users, queries, recommendations, and feedback.
        Use only for testing/development.
        """
        logger.warning("Deleting ALL recommendation data from database")

        try:
            with self.Session() as session:
                # Delete in correct order due to relationships
                feedback_count = session.query(RecommendationFeedbackModel).delete()
                rec_count = session.query(RecommendationModel).delete()
                query_count = session.query(QueryHistoryModel).delete()
                user_count = session.query(UserModel).delete()

                session.commit()
                logger.info(f"Successfully deleted recommendation data: "
                           f"{user_count} users, {query_count} queries, "
                           f"{rec_count} recommendations, {feedback_count} feedback")
        except Exception as e:
            logger.error(f"Failed to delete recommendation data: {e}")
            raise


