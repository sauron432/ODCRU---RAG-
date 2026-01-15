from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# print(BASE_DIR)
DATA_DIR = BASE_DIR/ "data/airlines_reviews.csv"

# print(DATA_DIR)

COLUMN_NAME = "Reviews"
CHUNK_SIZE = 250
OVERLAP_SIZE = 50
