import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

from src import logger

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    data_raw: Path = PROJECT_ROOT / "Data"
    vectorstore_dir: Path = PROJECT_ROOT / "Data" / "vectorstore"
    # default input file names (can be overridden via env)
    raw_recipes_csv: Path = PROJECT_ROOT / "Data" / "Food" / "RAW_recipes.csv"


@dataclass(frozen=True)
class Secrets:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")


@dataclass(frozen=True)
class Models:
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    gemini_model_name: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


@dataclass(frozen=True)
class Params:
    persist_directory: Path = Paths().vectorstore_dir
    top_k: int = int(os.getenv("TOP_K", "5"))
    generator_max_tokens: int = int(os.getenv("GEN_MAX_TOKENS", "700"))
    generator_temperature: float = float(os.getenv("GEN_TEMPERATURE", "0.2"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "32"))


@dataclass(frozen=True)
class DataSources:
    food_dataset_url: str = os.getenv(
        "FOOD_DATASET_URL",
        "https://drive.google.com/file/d/1M5c0pXG9onlrm0z0IlgAoFyz8C58kkFo/view?usp=sharing",
    )


paths = Paths()
secrets = Secrets()
models = Models()
params = Params()
data_sources = DataSources()

if __name__ == "__main__":
    paths.data_raw.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data directory set to: {paths.data_raw}")
    params.persist_directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Vectorstore directory set to: {params.persist_directory}")
