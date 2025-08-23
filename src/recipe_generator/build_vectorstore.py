import pandas as pd
from pathlib import Path
from .config import params, paths
from .vectorstore import build_chroma_from_dataframe
from .utils import combine_text_row
from .. import logger


def main(
    csv: str = str(paths.raw_recipes_csv),
    limit: int = None,
    persist_dir: str = str(params.persist_directory),
):
    """
    Build a Chroma vectorstore from a CSV of recipes.
    Parameters can be overridden when calling main().
    """

    csv_path = Path(csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    logger.info(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    # Ensure recipe_text exists
    if "recipe_text" not in df.columns:
        print("Creating 'recipe_text' from ingredients/steps/tags...")
        df = df.copy()
        df["recipe_text"] = df.apply(combine_text_row, axis=1)

    logger.info(f"Building Chroma (limit={limit})...")
    vs = build_chroma_from_dataframe(
        df=df,
        text_col="recipe_text",
        limit=limit,
        persist_directory=persist_dir,
    )

    logger.info(f"Vectorstore built and persisted at: {persist_dir}")
    return vs


if __name__ == "__main__":
    main()
