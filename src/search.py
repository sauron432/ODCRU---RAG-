import ollama
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity


def search(query:str, embeddings: np.ndarray, all_chunks: List[str]) -> str:
    """Takes query, embeds it and performs cosine similarity search and returns a response from the data source

    Args:
        query (str):Input query.

    Returns:
        str: Best match from the source.
    """
    if embeddings is None:
        print("No stored embeddings found!")
    try:
        print("------ Generating Response ------") 
        query_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=query)["embedding"]
        
    except Exception as e:
        print(f"Errpr embedding query: {e}")

    cos = cosine_similarity(
        [query_embedding],
        embeddings
        )[0]
    
    best_index = np.argmax(cos)
    return all_chunks[best_index]

