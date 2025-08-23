import json
from .generate_recipe import generate_recipe_from_request

user_request = "I need a quick dinner with chicken, garlic, cheese, high in protein, under 20 minutes, for 3 people. oh i also have bread."


def main():

    result = generate_recipe_from_request(user_request)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
