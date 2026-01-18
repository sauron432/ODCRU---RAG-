import ollama
import numpy as np
from typing import List

from src.config import MODEL

def get_embeddings(text_list) -> np.ndarray:
    
    print("------ Creating Embeddings ------")
    embeddings = []
    texts = [c["text"] for c in text_list]
    for text in texts:
        response = ollama.embed(MODEL, input=text)
        embeddings.append(response["embeddings"][0])
    return np.array(embeddings)
