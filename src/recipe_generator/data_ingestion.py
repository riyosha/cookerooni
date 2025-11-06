import os
import gdown
import zipfile
from src import logger
from src.recipe_generator.config import paths, data_sources


def download_and_extract_food_data(data_dir="Data"):
    dataset_url = data_sources.food_dataset_url
    zip_download_path = os.path.join(data_dir, "Food.zip")
    unzip_path = os.path.join(data_dir, "Food")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(unzip_path, exist_ok=True)

    logger.info(f"Downloading data from {dataset_url} into file {zip_download_path}")
    file_id = dataset_url.split("/")[-2]
    prefix = "https://drive.google.com/uc?/export=download&id="
    gdown.download(prefix + file_id, zip_download_path)

    try:
        with zipfile.ZipFile(zip_download_path, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"Successfully extracted zip file to {unzip_path}")
    except Exception as e:
        logger.error(f"Failed to extract zip file: {e}")


if __name__ == "__main__":
    download_and_extract_food_data()
