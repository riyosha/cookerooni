import json
from typing import Any, Dict, Optional
from langchain.chains import LLMChain

from .vectorstore import load_chroma
from .chains import create_recipe_chain, get_gemini_llm
from .config import params
from .utils import build_context_block
import pandas as pd


def _ensure_json_object(text: str) -> Dict[str, Any]:
    """
    Parse and lightly validate the LLM's JSON output.
    If the LLM returns extra text, try to find the first/last braces.
    """
    text = text.strip()
    # Find first '{' and last '}' without using regex
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    obj = json.loads(text)
    # Minimal schema checks
    for key in ["title", "total_minutes", "ingredients", "steps", "nutrition"]:
        if key not in obj:
            raise ValueError(f"Missing key in response JSON: {key}")
    return obj


def generate_recipe(
    user_ingredients: str, max_minutes: int, k: int = 20
) -> Dict[str, Any]:
    """
    Retrieve relevant recipes and synthesize a single recipe as structured JSON.
    """
    vector_db = load_chroma()
    retrieved_docs = vector_db.similarity_search(user_ingredients, k=k)

    # Build a small pandas frame to reuse context builder
    rows = []
    for d in retrieved_docs:
        m = d.metadata
        rows.append(
            {
                "name": m.get("name"),
                "minutes": m.get("minutes"),
                "ingredients": m.get("ingredients"),
                "steps": m.get("steps"),
                "nutrition": m.get("nutrition"),
            }
        )
    context_str = build_context_block(pd.DataFrame(rows))

    # Run LLM chain
    recipe_chain: LLMChain = create_recipe_chain()
    response_text: str = recipe_chain.run(
        {
            "context": context_str,
            "user_ingredients": user_ingredients,
            "max_minutes": max_minutes,
        }
    )

    return _ensure_json_object(response_text)
