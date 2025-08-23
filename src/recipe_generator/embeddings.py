from langchain_community.embeddings import HuggingFaceEmbeddings
from .config import models
from .. import logger


def get_embedding_model():
    """
    Returns a HuggingFaceEmbeddings instance.
    """
    logger.info(f"Using embedding model: {models.embedding_model_name}")
    return HuggingFaceEmbeddings(model_name=models.embedding_model_name)
