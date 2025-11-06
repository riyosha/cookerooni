from langchain.prompts import PromptTemplate, ChatPromptTemplate

SYSTEM_PROMPT = """You are Cookeroni, a careful, practical recipe assistant.
You ALWAYS base your answers on the provided recipe context and user ingredients.
You must obey all constraints and keep output JSON strictly valid (no comments or prose)."""

RECIPE_PROMPT: PromptTemplate = PromptTemplate(
    input_variables=[
        "context",
        "user_ingredients",
        "max_minutes",
        "servings",
        "nutrition",
    ],
    template="""
You will synthesize ONE recipe using the CONTEXT below.

CONSTRAINTS:
- Must use these user ingredients: {user_ingredients}
- Total time must be <= {max_minutes} minutes
- Must serve {servings} people
- Nutrition should match as closely as possible: {nutrition}
- If constraints are impossible, approximate but stay reasonable
- Output JSON only.

CONTEXT:
{context}

RESPONSE FORMAT (valid JSON):
{{
  "title": "...",
  "total_minutes": <int>,
  "servings": <int>,
  "ingredients": ["..."],
  "steps": ["1) ...", "2) ..."],
  "nutrition": {{
    "calories": <int>,
    "protein_g": <int>,
    "total_fat_g": <int>,
    "carbohydrates_g": <int>,
    "saturated_fat_g": <int>,
    "sugar_g": <int>,
    "sodium_g": <int>
  }}
}}
""",
)

PREFERENCE_PROMPT = ChatPromptTemplate.from_template(
    """
You are a parser. Extract structured recipe preferences from the user request. 
Always return valid JSON only.

USER REQUEST:
{user_request}

JSON FORMAT:
{{
  "ingredients": ["..."],
  "max_minutes": <int or null>,
  "servings": <int or null>,
  "nutrition": {{
      "calories": <int or null>,
      "protein_g": <int or "high"|"medium"|"low">,
      "saturated_fat_g": <int or null>,
      "total_fat_g": <int or null>,
      "carbohydrates_g": <int or null>,
      "sodium_g": <int or null>,
      "sugar_g": <int or null>
  }}
}}
"""
)
