import logging
from typing import List, Dict, Any
from pymilvus import (
    connections, utility, Collection, CollectionSchema, FieldSchema, DataType
)

# Get a logger for the current module
logger = logging.getLogger(__name__)

class VectorHandler:
    """
    Manages all interactions with the Milvus vector database, including
    connection, collection management, and vector searches.
    """
    def __init__(self, host: str, port: str, collection_name: str, vector_dim: int):
        """
        Initializes the connection to Milvus and ensures the collection exists.

        Args:
            host (str): The hostname or IP address of the Milvus server.
            port (str): The port number for the Milvus server.
            collection_name (str): The name of the collection to use.
            vector_dim (int): The dimension of the vectors to be stored.
        """
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        
        try:
            connections.connect(alias="default", host=host, port=port)
            logger.info(f"Successfully connected to Milvus at {host}:{port}")
            
            # The index metric is defined at collection creation. 
            # Defaulting to "IP" as it's common for normalized text embeddings.
            self._ensure_collection_exists(metric_type="IP")
            
            self.collection = Collection(self.collection_name)
            self.collection.load()
            logger.info(f"Vector collection '{self.collection_name}' is loaded and ready for searching.")
            
        except Exception as e:
            logger.error(f"Failed to connect to Milvus or load collection: {e}")
            raise

    def _ensure_collection_exists(self, metric_type: str):
        """
        Creates the collection with a specific index metric if it does not already exist.

        Args:
            metric_type (str): The metric to use for the index (e.g., 'IP', 'L2').
        """
        if utility.has_collection(self.collection_name):
            logger.debug(f"Collection '{self.collection_name}' already exists.")
            # Note: This does not check if the existing metric matches.
            # For a production system, a more robust check might be needed.
            return

        logger.info(f"Collection '{self.collection_name}' not found. Creating now with metric '{metric_type}'.")
        id_field = FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256)
        embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
        schema = CollectionSchema(fields=[id_field, embedding_field], description="Document chunk embeddings")
        
        self.collection = Collection(name=self.collection_name, schema=schema)
        index_params = {
            "metric_type": metric_type,
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        logger.info(f"Collection '{self.collection_name}' created successfully.")

    def add_embedding(self, chunk_id: str, vector: List[float]):
        """
        Adds a new vector embedding to the collection.

        Args:
            chunk_id (str): The unique ID of the chunk (must match the ID in the SQL DB).
            vector (List[float]): The embedding vector.
        """
        try:
            self.collection.insert([[chunk_id], [vector]])
            self.collection.flush()
            logger.debug(f"Inserted vector for chunk ID: {chunk_id}")
        except Exception as e:
            logger.error(f"Failed to insert vector for chunk ID {chunk_id}: {e}")
            raise

    def search_similar(self, query_vector: List[float], k: int, metric_type: str) -> List[Dict[str, Any]]:
        """
        Searches for the k most similar vectors using the specified metric.

        Args:
            query_vector (List[float]): The vector to search against.
            k (int): The number of similar results to return.
            metric_type (str): The metric to use for the search ('IP' or 'L2').

        Returns:
            List[Dict[str, Any]]: A list of search results, each containing 'id' and 'distance'.
        """
        logger.info(f"Performing search for {k} nearest neighbors with metric '{metric_type}'.")
        # Robustness check: Ensure collection is loaded before searching.
        try:
            if not self.collection.is_empty and not self.collection.has_partition("_default"):
                logger.warning(f"Collection '{self.collection_name}' was not loaded. Loading now...")
                self.collection.load()
        except Exception as e:
            logger.warning(f"Could not verify collection load status. Attempting to load regardless. Error: {e}")
            self.collection.load()

        search_params = {"metric_type": metric_type, "params": {"nprobe": 10}}
        
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["id"]
        )
        hits = results[0]
        logger.info(f"Search completed. Found {len(hits)} results.")
        # Note: The 'distance' field from Milvus will be a similarity score for 'IP'
        # and a distance for 'L2'.
        return [{"id": hit.id, "distance": hit.distance} for hit in hits]

    def get_vectors_by_ids(self, chunk_ids: List[str]) -> Dict[str, List[float]]:
        """
        Retrieves the full embedding vectors for a given list of chunk IDs.

        Args:
            chunk_ids (List[str]): The list of IDs to retrieve vectors for.

        Returns:
            Dict[str, List[float]]: A dictionary mapping each chunk ID to its vector.
        """
        if not chunk_ids:
            return {}
        logger.debug(f"Querying for {len(chunk_ids)} vectors by ID.")
        expr = f"id in {chunk_ids}"
        try:
            results = self.collection.query(expr=expr, output_fields=["id", "embedding"])
            logger.info(f"Successfully retrieved {len(results)} vectors.")
            return {res['id']: res['embedding'] for res in results}
        except Exception as e:
            logger.error(f"Failed to query vectors by ID: {e}")
            return {}

    def delete_all_data(self):
        """Deletes all data by dropping and recreating the collection."""
        logger.warning(f"Dropping collection '{self.collection_name}' to delete all data.")
        try:
            utility.drop_collection(self.collection_name)
            # Recreate it immediately with the default metric.
            self._ensure_collection_exists(metric_type="IP")
            logger.info(f"Collection '{self.collection_name}' has been successfully dropped and recreated.")
        except Exception as e:
            logger.error(f"Failed to delete and recreate collection '{self.collection_name}': {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        Gets statistics about the vector collection.

        Returns:
            Dict[str, Any]: A dictionary with the total number of vectors.
        """
        logger.debug("Getting vector database statistics.")
        try:
            self.collection.flush() # Ensure count is up-to-date
            return {"total_vectors": self.collection.num_entities}
        except Exception as e:
            logger.error(f"Failed to get vector database stats: {e}")
            return {"total_vectors": -1}