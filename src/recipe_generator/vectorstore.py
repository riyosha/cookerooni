from typing import List, Optional
import pandas as pd
from langchain.vectorstores import Chroma
from .embeddings import get_embedding_model
from .config import settings
from .utils import combine_text_row, to_metadata_records


def build_chroma_from_dataframe(
    df: pd.DataFrame,
    text_col: str = "recipe_text",
    limit: Optional[int] = None,
    persist_directory: Optional[str] = None,
) -> Chroma:
    """
    Build and persist a Chroma vectorstore from a dataframe of recipes.
    """
    persist_dir = str(persist_directory or settings.persist_directory)
    if "recipe_text" not in df.columns or (text_col != "recipe_text"):
        df = df.copy()
        df[text_col] = df.apply(combine_text_row, axis=1)

    if limit:
        df = df.head(limit)

    texts: List[str] = df[text_col].tolist()
    metadatas = to_metadata_records(df)

    embedding = get_embedding_model()
    vs = Chroma.from_texts(
        texts=texts,
        embedding=embedding,
        metadatas=metadatas,
        persist_directory=persist_dir,
    )
    vs.persist()
    return vs


def load_chroma(persist_directory: Optional[str] = None) -> Chroma:
    """
    Load an existing Chroma vectorstore from disk.
    """
    persist_dir = str(persist_directory or settings.persist_directory)
    embedding = get_embedding_model()
    return Chroma(persist_directory=persist_dir, embedding_function=embedding)
