from .vectorstore import load_chroma
from .. import logger

# Set your query and number of results here
query = "chicken, butter, garlic"
k_results = 5


def main():
    vs = load_chroma()
    docs = vs.similarity_search(query, k=k_results)
    for i, d in enumerate(docs, 1):
        meta = d.metadata
        print(f"[{i}] {meta.get('name')} ({meta.get('minutes')} min)")
        ing = meta.get("ingredients") or []
        if isinstance(ing, str):
            ing = [ing]
        print("    Ingredients:", ", ".join(ing[:12]), ("..." if len(ing) > 12 else ""))

        tags = meta.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        print("    Tags:", ", ".join(tags[:10]))


if __name__ == "__main__":
    main()
