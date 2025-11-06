import json
from .generate_recipe import generate_recipe_from_request

DEFAULT_REQUEST = "I need a quick dinner with chicken, garlic, cheese, high in protein, under 20 minutes, for 3 people. oh i also have bread."


def main():
    user_request = input("Enter your recipe request (or press enter to use default): ")
    if not user_request.strip():
        user_request = DEFAULT_REQUEST
    result = generate_recipe_from_request(user_request)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
