import ollama
import numpy as np
from typing import List

def get_embeddings(text_list: List[str]) -> np.ndarray:
    """Generated embeddings for a list of strings and returns a numpy array of embeddings 

    Args:
        text_list (List[str]): chunks received after chunking

    Returns:
        np.array: an array of embeddings of the chunks
    """
    print("------ Creating Embeddings ------")
    embeddings = []
    for text in text_list:
        response = ollama.embed(model="mxbai-embed-large", input=text)
        embeddings.append(response["embeddings"][0])
    return np.array(embeddings)