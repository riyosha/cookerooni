from langchain.prompts import PromptTemplate

SYSTEM_PROMPT = (
    "You are Cookeroni, a careful, practical recipe assistant.\n"
    "You ALWAYS base your answers on the provided recipe context and user ingredients.\n"
    "You must obey all constraints and keep output JSON strictly valid (no comments or prose)."
)

RECIPE_PROMPT: PromptTemplate = PromptTemplate(
    input_variables=["context", "user_ingredients", "max_minutes"],
    template="""
You will synthesize ONE recipe using the CONTEXT below.

CONSTRAINTS:
- Use only the user ingredients when possible: {user_ingredients}
- Total time must be <= {max_minutes} minutes
- Include title, ingredients, steps, nutrition info

CONTEXT:
{context}

RESPONSE FORMAT (valid JSON):
{{
  "title": "...",
  "total_minutes": <int>,
  "ingredients": ["..."],
  "steps": ["1) ...", "2) ...", "..."],
  "nutrition": {{"calories": <int>, "protein_g": <int>, "fat_g": <int>, "carbs_g": <int>, "source": "context|approx"}}
}}
""".strip(),
)
