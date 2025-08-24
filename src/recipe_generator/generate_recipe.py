import json
from typing import Any, Dict, Optional
from langchain.chains import LLMChain

from .vectorstore import load_chroma
from .chains import create_recipe_chain, create_preference_parsing_chain, get_gemini_llm
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


def parse_preferences(user_request: str):
    """
    Parses a natural language user request into structured JSON.
    """
    preference_chain: LLMChain = create_preference_parsing_chain()
    raw_json = preference_chain.run({"user_request": user_request})
    if raw_json.strip().startswith("```"):
        raw_json = raw_json.strip().strip("`").strip("json").strip()
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON returned by LLM: {raw_json}")


def generate_recipe_from_request(
    user_request: str,
    k: int = 20,
    max_minutes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Natural-language RAG:
    1) Parse user preferences from request
    2) Retrieve relevant recipes from vectorstore
    3) Build context from neighbors
    4) Generate JSON recipe with LLM
    """
    vector_db = load_chroma()
    prefs = parse_preferences(user_request)
    ingredients = prefs.get("ingredients", [])
    minutes = prefs.get("max_minutes", max_minutes if max_minutes is not None else 60)
    servings = prefs.get("servings", 2)
    nutrition = prefs.get("nutrition", {})

    docs = vector_db.similarity_search(", ".join(ingredients), k=k)
    rows: list = []
    for d in docs:
        m = d.metadata or {}
        rows.append(
            {
                "name": m.get("name"),
                "minutes": m.get("minutes"),
                "ingredients": m.get("ingredients"),
                "steps": m.get("steps"),
                "nutrition": m.get("nutrition"),
            }
        )
    context = build_context_block(pd.DataFrame(rows))

    recipe_chain: LLMChain = create_recipe_chain()
    response_text: str = recipe_chain.run(
        {
            "context": context,
            "user_ingredients": ", ".join(ingredients),
            "max_minutes": minutes,
            "servings": servings,
            "nutrition": nutrition,
        }
    )

    return _ensure_json_object(response_text)
