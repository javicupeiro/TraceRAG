import logging
from sentence_transformers import SentenceTransformer
from typing import List

# Get a logger for the current module
logger = logging.getLogger(__name__)

class Embedder:
    """
    A wrapper class for a sentence-transformer model to handle text embeddings.
    
    This class abstracts the model loading and embedding process, making it easy
    to swap out embedding models in the future.
    """
    def __init__(self, model_name: str):
        """
        Initializes the Embedder and loads the specified sentence-transformer model.

        Args:
            model_name (str): The name of the sentence-transformer model from Hugging Face
                              (e.g., 'all-MiniLM-L6-v2').
        
        Raises:
            Exception: If the model fails to load.
        """
        logger.debug(f"Attempting to initialize Embedder with model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Embedder initialized successfully with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{model_name}': {e}")
            raise

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Creates vector embeddings for a list of text strings.

        Args:
            texts (List[str]): A list of strings to be embedded.

        Returns:
            List[List[float]]: A list of embedding vectors, where each vector is a list of floats.
        
        Raises:
            Exception: If the embedding process fails.
        """
        if not texts:
            logger.warning("Embed method called with an empty list of texts.")
            return []
            
        logger.info(f"Creating embeddings for {len(texts)} text(s).")
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False).tolist()
            logger.info(f"Successfully created {len(embeddings)} embeddings.")
            return embeddings
        except Exception as e:
            logger.error(f"An error occurred during the embedding process: {e}")
            raise