from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# print(BASE_DIR)
DATA_DIR = BASE_DIR/ "data/airlines_reviews.csv"
MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3.2"
USER = "default-user"
TEMP = 0.2
# print(DATA_DIR)

COLUMN_NAME = "Reviews"
CHUNK_SIZE = 500
OVERLAP_SIZE = 50
AIRLINE = "Airline"
