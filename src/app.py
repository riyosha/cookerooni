from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import json

from src.recipe_generator.generate_recipe import generate_recipe_from_request
from src import logger

app = FastAPI(
    title="Cookerooni Recipe Generator",
    description="Generate personalized recipes using AI and RAG",
    version="1.0.0",
)


class RecipeRequest(BaseModel):
    prompt: str
    k: int = 20
    max_minutes: Optional[int] = None


class RecipeResponse(BaseModel):
    title: str
    total_minutes: int
    servings: int
    ingredients: list[str]
    steps: list[str]
    nutrition: dict


@app.post("/generate", response_model=RecipeResponse)
async def generate_recipe(request: RecipeRequest):
    """
    Generate a recipe based on user prompt and preferences.

    - prompt: Natural language description of what you want to cook
    - k: Number of similar recipes to consider (default: 20)
    - max_minutes: Maximum cooking time in minutes (optional)
    """
    try:
        logger.info(f"Received recipe request: {request.prompt}")
        result = generate_recipe_from_request(
            user_request=request.prompt,
            k=request.k,
            max_minutes=request.max_minutes,
        )
        logger.info("Recipe generated successfully")
        return result
    except Exception as e:
        logger.error(f"Error generating recipe: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error generating recipe: {str(e)}"
        )


@app.get("/")
def read_root():
    return {
        "message": "Welcome to Cookerooni! 🍳",
        "description": "AI-powered recipe generator using RAG",
        "docs": "/docs",
        "endpoints": {"generate": "POST /generate - Generate a recipe from a prompt"},
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "cookerooni-api"}


# Use command uvicorn src.app:app --reload --host 127.0.0.1 --port 8000 to run the server

# Use:
# curl -X POST "http://127.0.0.1:8000/generate" \
#  -H "Content-Type: application/json" \
#  -d '{"prompt": "Give me a quick recipe with kidney beans, tomatoes, coriander, onions, rices and some spices."}'
# to generate recipe
