import chromadb
import os
from chromadb.config import Settings


def create_vectorDB():
    directory = '../data/vector_store'
    os.makedirs(directory, exist_ok=True)

    client = chromadb.Client(
        Settings(
            persist_directory=directory,
            anonymized_telemetry=False,
            is_persistent=True)
    )
    collection = client.get_or_create_collection(
        name="airline_reviews"
    )
    return collection