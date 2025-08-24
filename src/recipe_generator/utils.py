from typing import Any, List, Dict
import ast
import pandas as pd


def safe_literal_list(x: Any) -> List[Any]:
    """
    Safely convert a stringified Python list to a real list using ast.literal_eval.
    Returns [] on failure or NaN.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return x
    try:
        val = ast.literal_eval(str(x))
        return val if isinstance(val, list) else []
    except Exception:
        return []


def combine_text_row(row: pd.Series) -> str:
    """
    Combine ingredients, steps, and tags into a single text blob for embedding.
    """
    ingredients = " ; ".join(safe_literal_list(row.get("ingredients")))
    steps = " ".join(safe_literal_list(row.get("steps")))
    tags = "; ".join(safe_literal_list(row.get("tags")))
    return f"{ingredients} {steps} {tags}".strip()


def build_context_block(df_rows: pd.DataFrame) -> str:
    """
    Turn recipe rows into readable blocks for the LLM context.
    """
    blocks: List[str] = []
    for _, r in df_rows.iterrows():
        ing = ", ".join(safe_literal_list(r.get("ingredients")))
        steps_list = safe_literal_list(r.get("steps"))
        steps_joined = " ".join(steps_list)[:1500] + ("..." if steps_list else "")
        block = (
            f"TITLE: {r.get('name','')}\n"
            f"MINUTES: {r.get('minutes','')}\n"
            f"INGREDIENTS: {ing}\n"
            f"STEPS: {steps_joined}\n"
            f"NUTRITION_RAW: {r.get('nutrition','')}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def to_metadata_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert a dataframe subset to a list of metadata dicts for vectorstore.
    Ensures list-typed columns are stored as Python lists (not strings).
    """
    records = []
    for _, r in df.iterrows():
        records.append(
            {
                "name": r.get("name"),
                "ingredients": safe_literal_list(r.get("ingredients")),
                "steps": safe_literal_list(r.get("steps")),
                "minutes": r.get("minutes"),
                "tags": safe_literal_list(r.get("tags")),
                "nutrition": r.get("nutrition"),
                "id": r.get("id", None),
            }
        )
    return records
