from typing import List, Optional, Dict, Any
import pandas as pd
from langchain_community.vectorstores import Chroma
from tqdm import tqdm
from .embeddings import get_embedding_model
from .config import params
from .utils import combine_text_row, safe_literal_list
from .. import logger


def to_metadata_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    records = df.to_dict(orient="records")
    for r in records:
        r["ingredients"] = ", ".join(safe_literal_list(r.get("ingredients")))
        r["steps"] = " ".join(safe_literal_list(r.get("steps")))
        r["tags"] = "; ".join(safe_literal_list(r.get("tags")))
    logger.info("Metadata records conversion complete.")
    return records


def build_chroma_from_dataframe(
    df: pd.DataFrame,
    text_col: str = "recipe_text",
    limit: Optional[int] = None,
    persist_directory: Optional[str] = None,
    batch_size: int = params.batch_size,
) -> Chroma:
    """
    Build and persist a Chroma vectorstore from a dataframe of recipes.
    """
    logger.info(f"Building Chroma vectorstore from dataframe with {len(df)} rows.")
    persist_dir = str(persist_directory or params.persist_directory)
    if "recipe_text" not in df.columns or (text_col != "recipe_text"):
        df = df.copy()
        df[text_col] = df.apply(combine_text_row, axis=1)

    if limit:
        logger.info(f"Limiting dataframe to first {limit} rows.")
        df = df.head(limit)

    texts: List[str] = df[text_col].tolist()
    metadatas = to_metadata_records(df)

    # Initialize embedding model
    logger.info("Initializing embedding model for Chroma.")
    embedding = get_embedding_model()

    # Create Chroma vectorstore
    logger.info(f"Creating Chroma vectorstore at '{persist_dir}'.")
    vs = Chroma.from_texts(
        texts=texts,
        embedding=embedding,
        metadatas=metadatas,
        persist_directory=persist_dir,
    )

    vs.persist()
    logger.info(f"Chroma vectorstore persisted at '{persist_dir}'.")
    return vs


def load_chroma(persist_directory: Optional[str] = None) -> Chroma:
    """
    Load an existing Chroma vectorstore from disk.
    """
    logger.info("Loading Chroma vectorstore from disk.")
    persist_dir = str(params.persist_directory)
    embedding = get_embedding_model()
    logger.info(f"Chroma vectorstore loaded from '{persist_dir}'.")
    return Chroma(persist_directory=persist_dir, embedding_function=embedding)
