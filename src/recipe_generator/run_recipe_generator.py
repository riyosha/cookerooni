import json
from .generate_recipe import generate_recipe

ingredients = "chicken, butter, garlic"
max_minutes = 15
k_results = 20


def main():
    result = generate_recipe(ingredients, max_minutes, k=k_results)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
