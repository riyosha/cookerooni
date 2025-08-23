from langchain.embeddings import HuggingFaceEmbeddings
from .config import models


def get_embedding_model():
    """
    Returns a HuggingFaceEmbeddings instance.
    """
    return HuggingFaceEmbeddings(model_name=models.embedding_model_name)
